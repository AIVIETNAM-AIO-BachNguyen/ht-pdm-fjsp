# Can thiệp riêng technician đầu: chưa có căn cứ tránh việc dài

**Chẩn đoán không ủng hộ giải thích “candidate thua vì không chờ thay cho việc
4 tick”.** Giữ nguyên candidate ở các bước sau, đổi riêng lựa chọn đầu sang
WAIT tại 9 tình huống này làm cost tăng trung bình 1,916667/máy và failure tăng
0,5/máy. Không triển khai duration penalty, luật né việc dài hay GO-head mới từ
kết quả này. Vòng FULL trước vẫn âm tính 2/16, mục tiêu học từ đầu chưa đạt.

## Đã chạy và kiểm chứng

- Protocol fbf6c94; code frozen 581f9c98c5605fef6403694ab23e47dd5ce62fe3.
- 12 tests PASS trong 1,54 giây; commit/push trước chạy trực tiếp trên lab.
- Artifact `technician_capacity_prefix_counterfactual_v1_20261016T073809934365944Z`:
  COMPLETED/audit PASS, 40,378285 giây, 576 nhánh, gồm 144 no-op và 432 can thiệp.
- Chỉ dùng 144 trạng thái candidate đã lưu: 48 trạng thái × 3 training replicas,
  physical DEV seed 230100000 đã mở. Không seed vật lý mới, không training,
  không scientific test, không teacher labels hoặc pretraining bank.
- Mỗi nhánh sao chép toàn bộ simulator và cùng event tape. Chỉ đổi T1 ở đúng
  một thời điểm; T2 được candidate tính lại theo reservation/calendar thực tế;
  các bước sau vẫn candidate, có cùng deadline tail 12 ticks.
- No-op tái hiện bit-exact toàn bộ quan sát còn lại và mọi final source metric
  với sai số tối đa 1e-8. Source bindings 548 mục được kiểm tra trước/sau.

Verifier độc lập ed039c5 kiểm lại hash của 548 source bindings và 7 diagnostic
outputs, đủ 576 dòng, legality/token/duration, chênh lệch cost/health, strata,
mọi replica và verdict: PASS. Nó kiểm độc lập dữ liệu/phép tính; không chạy lại
simulator độc lập. Artifact verification:
`technician_capacity_prefix_counterfactual_v1_verification_20261016T074055778312738Z`.

## Kết quả mô tả

Gain dương nghĩa là đổi lựa chọn làm giảm final episode cost; chia cho 4 máy.
Health delta là alternative trừ candidate. Đây là hiệu ứng của MỘT can thiệp
tại từng trạng thái, rồi tiếp tục bằng cùng policy; không cộng các gain lại và
không coi trung bình này là hiệu năng của một policy mới.

| Stratum | Nhánh T1 | Số context | Mean gain/máy | Tốt hơn / kém hơn / bằng |
|---|---|---:|---:|---|
| Tất cả | WAIT |144|2,147569|45 / 34 / 65|
| Tất cả | Shortest legal |144|−0,053819|4 / 4 / 136|
| Tất cả | Lựa chọn đầu của flat |144|−0,189236|7 / 4 / 133|
| Candidate chọn 4 ticks | WAIT |9|−1,916667|5 / 4 / 0|
| Candidate chọn 4 ticks | Shortest legal |9|−1,305556|0 / 1 / 8|
| Candidate chọn 4 ticks | Lựa chọn đầu của flat |9|0,916667|4 / 1 / 4|
| Candidate chọn 1 hoặc 2 ticks | WAIT |74|4,412162|40 / 30 / 4|
| Candidate WAIT | WAIT |61|0|0 / 0 / 61|

Shortest thay T1 token ở 14/144 context; flat thay ở 11/144; giữ đầy đủ các
nhánh không thay đổi để tránh chỉ báo cáo những can thiệp có lợi. 144 no-op đều
có gain/health delta bằng 0.

Ở 9 context 4 ticks, WAIT cải thiện availability trung bình 0,010995 nhưng làm
failure/máy tăng 0,5; pending/máy không đổi. Ba replica lần lượt có 1/4/4 context
và mean gain −22,75 / −3,75 / +5,125. Vì vậy không có lợi ích nhất quán giữa các
replica; availability tốt hơn không đủ khi failure và cost tăng.

## Tiêu chí kỹ thuật đã khóa

| Điều kiện cho giải thích deferral | Ngưỡng | Kết quả | Đạt |
|---|---|---:|---|
| Số context 4 ticks, đủ cả 3 replica |≥6|9, đủ 3|Có|
| WAIT mean cost gain/máy |≥1|−1,916667|Không|
| Mean gain dương ở mọi replica |Cả 3|Chỉ 1|Không|
| Availability delta |≥−0,01|+0,010995|Có|
| Pending delta/máy |≤+0,025|0|Có|
| Failure delta/máy |≤+0,25|+0,5|Không|

Không đạt đủ điều kiện; deferral_explanation_supported=false. Đây là gate kỹ
thuật mô tả, không phải hypothesis test hoặc bằng chứng trên quần thể. Các
trạng thái phụ thuộc trong 3 trajectory và chung một realized physical tape;
không có CI/p-value. Intervention dùng tương lai trong evaluator để đo hậu quả,
policy chỉ nhìn quan sát hiện tại; không dùng hậu quả làm dữ liệu huấn luyện.

## Hệ quả cho lượt tiếp theo

Hai giải thích đơn giản đã không được dữ liệu ủng hộ: T2 GO quá dễ trong cùng
prefix, và T1 cứ chọn việc dài là sai. Quan sát duration khác nhau không đủ để
thêm luật. Kết quả WAIT trên mọi trạng thái cũng không biện minh cho luôn WAIT:
các can thiệp chồng lấp và có cả tổn thất, còn backlog của nhóm T1 1/2 ticks tăng
0,030405/máy, vượt biên +0,025 nếu áp biên cũ cho phép so sánh mô tả này.

Trước khi thêm thành phần hoặc tăng budget, cần kiểm tra tín hiệu học return /
value trên dữ liệu phù hợp. Critic được học dưới stochastic training policy;
không được so với greedy DEV return rồi tự kết luận critic học sai hay GAE sẽ
giải quyết. Bất kỳ lượt train tiếp theo cần protocol riêng, random initialization,
đối chứng cùng thông tin/budget và các giới hạn sức khỏe ban đầu. Scientific
confirmation và larger-N vẫn đóng. Hiện không có full training đang chạy.

Sao chép thủ công về Mac khi cần; agent không thực hiện transfer:

```bash
mkdir -p "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results" &&
rsync -avhP --partial \
  bachng@100.111.83.52:~/ht-pdm-fjsp/artifacts/ \
  "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/"
```
