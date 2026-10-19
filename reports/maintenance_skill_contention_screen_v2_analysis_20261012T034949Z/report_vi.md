## Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent, inline.
- Origin Mode: analysis and read-only reconstruction.
- Origin Date: 2026-10-12.
- Verification Status: **ANALYZED** for interpretation; **VERIFIED** for artifact,
  metric-oracle, episode replay and registered CI reconstruction described below.
- Version Label: maintenance_skill_contention_screen_v2.

# Kết quả v2: tranh chấp kỹ năng có thật, độ khó phân công còn hạn chế

**Winner `overlap3__common_1_3` đạt cả bốn gate xác nhận đã đăng ký.** Đây là
tranh chấp technician có dịch vụ tốt hơn thật, không còn do phá hòa. Tuy nhiên,
kiểm tra bổ sung cho thấy mọi lựa chọn đổi technician trong các lượt tranh chấp
đều nằm giữa các máy tương đương về trạng thái và đặc tính vật lý. Vì vậy,
chưa nên chốt winner làm benchmark để chứng minh cải tiến assignment của RL.

## Tính toàn vẹn và phạm vi

Run `maintenance_skill_contention_screen_v2_full_cpu_20261012T034652Z` hoàn tất
trên CPU Linux trong61,05s, đúng commit sạch
`912ee749a556006e0164f7734450ccba9ef45bcb`. Đủ9.600episode screening và1.280episode
confirmation, tổng **10.880episode /652.800transition**; payload26.978.956bytes.

Kiểm tra **PASS**: 16hash artifact; sáu source snapshot khớp commit; protocol,
seed registry và75cấu hình khớp bản khóa; seed audit; episode grid;1.152row đối
chứng;19.200transition có log replay khớp. Tái dựng toàn bộ10.880episode cho mọi
metric, cost và terminal state đều khớp. Một oracle matching viết độc lập kiểm
tra primary/joint/cardinality trên **522.240trạng thái đo**; ranking và các CI
xác nhận tái tính độc lập đều khớp. Dữ liệu gốc không bị sửa.

Đây là screening môi trường dưới hai rule; chưa huấn luyện RL. Mỗi cấu hình cố
định4máy/2technician, đo48tick và12tick continuation. Screening dùng64seed;
confirmation dùng128seed mới. Inference chỉ áp dụng các trajectory mới ở cấu
hình cố định và các rule này, không phải mọi policy hay mọi kích thước.

## Setting được chọn

Có **2/72ứng viên** đạt gate screening. Winner là `overlap3__common_1_3`,
runner-up `overlap3__common_2_3`. Ba setting khác là đối chứng không được chọn.

| Máy | Technician 1 | Technician 2 |
|---|---|---|
| 1 | 1tick, phục hồi75% | Không đủ skill |
| 2 | 1tick, phục hồi75% | 3tick, phục hồi75% |
| 3 | 1tick, phục hồi75% | 3tick, phục hồi75% |
| 4 | 1tick, phục hồi75% | 3tick, phục hồi75% |

Technician 1 nhanh hơn trên mọi cạnh được phép; ba máy chung đều muốn người này.
Trong model, phục hồi75% nghĩa là tuổi máy sau hoàn tất bằng
`floor(tuổi trước sửa × 0,25)`, không phải xác suất sửa thành công75%.

Metric mới giữ mọi cạnh không bị dịch vụ khác trội hơn về cả thời gian và mức
phục hồi. Nếu dùng mọi cạnh hợp lệ phục vụ được Fmáy nhưng chỉ dùng các cạnh
tốt nhất phục vụ được P<Fmáy, một máy phải nhận dịch vụ kém hơn để phục vụ được
nhiều máy đồng thời. Primary vì thế không đếm thiếu tổng công suất đơn thuần.
Joint thêm yêu cầu cùng tập máy có nhiều phân công khác nhau về dịch vụ vật lý.

## Xác nhận theo protocol gốc: CONFIRMED

| Gate | Ước lượng bất lợi hơn giữa hai rule | CI98,75% | Kết quả |
|---|---:|---:|---|
| Tranh chấp kỹ năng | **12,89%** | 11,21–14,50% | Biên dưới≥10%: đạt |
| Tranh chấp kèm lựa chọn cùng lượt | **8,20%** | 6,92–9,49% | Biên dưới≥5%: đạt |
| Availability | **70,29%** | 69,71–70,89% | Biên dưới≥25%: đạt |
| Máy hỏng chưa được phục vụ tại H | **0,0313** | 0–0,0781 | Biên trên≤1: đạt |

Mẫu số của các tỷ lệ là toàn bộ48tick mỗi episode. Tương đương khoảng
**6,19lượt tranh chấp và3,94lượt tranh chấp kèm lựa chọn mỗi episode**. Có lựa
chọn vật lý nói chung44,01%, utilization59,42%, thiếu tổng công suất3,56%,
skill deficit0%. Availability bao gồm downtime vì đang PM/CM.

Bootstrap20.000lần theo seed, giữ hai rule ghép cặp; mỗi gate dùng CI98,75% theo
hiệu chỉnh bốn gate của protocol. Không đổi ranking hay chọn thay winner sau khi
đã mở confirmation. CI mô tả metric đã đăng ký, không chứng nhận độ khó RL.

Trong screening, `bottleneck2__common_2_4` có primary cao hơn14,29%, nhưng joint
chỉ2,05%; `overlap3__common_2_4` có13,77%/1,79%. Chúng không đủ cơ hội phân công
để đạt gate. Winner mạnh nhất **trong nhóm đạt gate của lưới đã xét**, không phải
cực đại toàn cục của conflict.

Đối chứng bằng nhau và tradeoff đều có primary/joint0%, dù legacy conflict của
chúng lần lượt25,59% và31,84–35,09%. Đối chứng partitioned cũng có primary/joint0%.
Điều này xác nhận metric mới loại được lỗi tie-break và tách được compatibility
bottleneck; tradeoff vẫn có thể có giá trị dài hạn dù không bị strict dominance.

## Hạn chế quan trọng: lựa chọn còn đối xứng

Ở winner, matching và greedy có **cùng action, cùng state, cùng trajectory** trên
tất cả128seed xác nhận, cả48tick đo và12tick continuation. Đã đối chiếu7.680bước
ghép cặp; đây không chỉ là trung bình trùng nhau. Cả hai rule có cost497,3594;
chênh lệch quan sát bằng0. Hai tên rule không tạo hai hành vi khác biệt trên panel
này; không coi chúng là hai bằng chứng độc lập.

Diagnostic hậu nghiệm trên6.144tick của128episode, chỉ đếm một rule vì hai path
trùng nhau:

- Có792tick strict competition và504tick joint choice.
- Trên504tick joint, kiểm tra **598tập máy** có phân công tối đa khác nhau về vật
  lý: **0/598tập có máy khác nhau** về tuổi, trạng thái hỏng, thời gian chờ, hàng
  service-time hoặc restoration. Cả504tick không có lựa chọn giữa máy khác tính.
- Với tập máy thực sự được rule chọn,439tick vừa strict vừa đổi được technician;
  **439/439tập gồm các máy tương đương**. Chênh lệch504và439do joint đếm tồn tại
  lựa chọn trên một tập máy tối đa, không bắt buộc rule chọn chính tập đó.

Ví dụ: hai máy chung đều khỏe, tuổi3, không chờ, cùng ma trận dịch vụ(1,3), cùng
mức phục hồi. Giao người nhanh cho máy A hoặc B đều có một máy mất1tick và máy
kia mất3tick; tổng thời gian dịch vụ vẫn4tick. Dưới cấu hình chi phí/failure hiện
tại, hai máy này có thể hoán đổi danh tính trong mô hình. **Suy luận từ đối xứng**:
lựa chọn giữa chúng chưa thể hiện ưu tiên kinh tế khác nhau. Một shock tape cố
định vẫn có thể cho realized cost khác nhau khi đổi phân công; không khẳng định
mọi counterfactual có cost y hệt.

Thước đo đã đăng ký không sai: nó xác nhận đổi dịch vụ từng máy. Nhưng tiêu chí
“khác dịch vụ vật lý” còn yếu nếu mục tiêu là quyết định **máy nào nên nhận người
giỏi hơn để giảm tổng cost**. Kết quả này không chứng minh mọi lựa chọn machine,
WAIT, timing hoặc mọi policy RL đều không có lợi ích.

Runner-up2vs3 có cost matching774,39so với greedy810,15: chênh lệch thăm dò4,41%,
CI95%3,52–5,30%, chưa hiệu chỉnh nhiều so sánh. Tuy nhiên, joint confirmation
thấp hơn giữa hai rule là4,93%, dưới ngưỡng5%. Nó không được xác nhận làm winner
và không được chọn thay sau khi xem test. Rule gap cũng gộp chọn máy và phân công,
không đo riêng assignment hoặc RL gain.

## Kết luận thực dụng

V2 đã xác nhận một setting có **tranh chấp technician nhanh thực sự**, khỏe hơn
setting v1 và không phụ thuộc phá hòa. Mình giữ `CONFIRMED` đúng phạm vi protocol,
nhưng **chưa chốt winner làm benchmark cải tiến assignment RL** vì các lựa chọn
tranh chấp còn đối xứng.

Bước tiếp theo nên phá đối xứng theo kỹ năng: để độ mất mát khi dùng technician
chậm khác nhau theo máy. Ví dụ giữ mask hiện tại, technician nhanh1tick; technician
chậm trên ba máy chung lần lượt3,5,2tick. Khi đó dành người nhanh cho máy có slow5
có ý nghĩa khác với dành cho máy có slow2. Đây chỉ là **đề xuất follow-up**, chưa
chạy, chưa chọn làm setting cuối và chưa có kết quả xác nhận. Metric bổ sung cần
kiểm tra joint choice giữa các máy không tương đương về trạng thái/ma trận dịch vụ,
và xác nhận lại trên panel mới trước khi đưa ra claim về assignment RL.

Không sửa protocol, không khởi chạy experiment mới trong lượt phân tích này.

## Fallacy scan:11/11

1. Simpson: giữ hai rule riêng; báo rõ chúng trùng trajectory ở winner.
2. Ecological: tỷ lệ episode không suy ra mọi state hay policy.
3. Berkson:72ứng viên thiết kế không đại diện không gian setting toàn cục.
4. Collider: không lọc episode theo cost, khả dụng hay thành công.
5. Base rate: all48tick; phân biệt504tick,598subset và439selected tick.
6. Regression to mean:128seed mới xác nhận winner, không dùng smoke làm test.
7. Survivorship: giữ đủ fail/pending/tail và mọi episode.
8. Look-elsewhere: screening thăm dò; bốn gate có hiệu chỉnh; symmetry và cost
   đánh dấu hậu nghiệm, không thay verdict gốc.
9. Forking paths: không rerank hoặc hạ gate sau khi mở confirmation.
10. Correlation/causation: rule cost gap không chứng minh lợi ích riêng assignment;
    diễn giải đối xứng là suy luận từ cấu trúc, không counterfactual cost experiment.
11. Reverse causality: cấu hình/rule khóa trước outcomes; không dùng outcomes
    để tuyên bố cơ chế riêng chưa được kiểm tra.

## Artifact phân tích

`analyze.py`, `analysis.json`, `validation.log`: kiểm tra/replay/metric oracle,
ranking, CI và diagnostic selected subset. `choice_symmetry.py`,
`choice_symmetry.json`, `choice_symmetry.log`: kiểm tra mọi tập máy joint choice.
Script dùng checkout commit ràng buộc tại
`/private/tmp/marl-maintenance-assignment-confidence-v1`; dữ liệu gốc vẫn ở
`lab_results/maintenance_skill_contention_screen_v2_full_cpu_20261012T034652Z`.
