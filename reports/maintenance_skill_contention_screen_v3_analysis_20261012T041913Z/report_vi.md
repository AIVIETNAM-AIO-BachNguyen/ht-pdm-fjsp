## Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent, inline.
- Origin Mode: analysis and read-only reconstruction.
- Origin Date: 2026-10-12.
- Verification Status: **ANALYZED** cho diễn giải; **VERIFIED** cho hash, source binding, replay, oracle metric và tái dựng selection mô tả dưới đây.
- Version Label: maintenance_skill_contention_screen_v3.

# V3: có phân công làm thay đổi downtime, chưa có winner theo protocol

**Không có ứng viên nào trong 48 setting đạt đủ năm gate screening.** Đây không phải lỗi chạy: run hoàn tất, audit PASS và kết quả tái dựng khớp. Vì không có winner, panel 128 seed mới chỉ đánh giá năm đối chứng; không có setting ứng viên nào được xác nhận trên panel này.

Điểm tiến bộ so với v2: đã thấy các cặp máy mà đổi technician làm thay đổi tổng thời gian sửa, kể cả trong hành động thực tế. Nhưng hai nhóm có vấn đề khác nhau: `bottleneck2` có nhiều cơ hội trong không gian hành động nhưng ít chọn chúng; `overlap3` thực sự chọn các cặp đổi người được, song tần suất dưới rule matching tốt hơn thấp hơn ngưỡng đã khóa.

## Tính toàn vẹn

Run: `maintenance_skill_contention_screen_v3_full_cpu_20261012T041544Z`, CPU Linux, 48,56 giây. Commit sạch: `19fb84d5d63977b8486afb6dde742b8864dd72ab`.

- Đủ 6.784 episode screening: 53 setting × 2 rule × 64 seed; trong đó 48 ứng viên và 5 đối chứng.
- Đủ 1.280 episode confirmation: chỉ 5 đối chứng × 2 rule × 128 seed mới.
- Tổng **8.064 episode / 483.840 transition**. Mỗi episode đo 48 tick và tiếp tục 12 tick dưới cùng rule tail.
- 19 hash artifact khớp; bảy source snapshot, hai registry lịch sử, protocol và registry hiện tại khớp commit; seed audit, episode grid và selection seal khớp.
- Replay 13.920 transition có journal thuộc 232 episode khớp state, event, action, metric, cost.
- Tái dựng **toàn bộ 8.064 episode** khớp mọi trường CSV. Oracle độc lập, đệ quy theo technician thay vì dùng bộ liệt kê của runner, kiểm tra **387.072 trạng thái đo**: matching tối đa, matching chỉ dùng cạnh không bị trội, physical choice, chênh lệch chi phí trên cùng nhóm máy và các rate mới đều khớp.
- Ranking, rejection và selection tái tính khớp. Đối chứng PASS trên 1.920 row. Không sửa dữ liệu gốc hoặc code thí nghiệm.

## Các setting đáng chú ý

Các tỷ lệ dưới đây lấy **rule thấp hơn** trong hai rule, mẫu số là toàn bộ 48 tick mỗi episode, không chỉ những tick có tranh chấp.

| Setting | Tranh chấp kỹ năng | Có nhóm máy đổi người làm đổi cost | Nhóm máy thực sự chọn có đổi cost | Gate không đạt |
|---|---:|---:|---:|---|
| Ngưỡng khóa trước run | ≥10% | ≥5% | ≥3% | — |
| `bottleneck2__fast2_slow643` | 14,19% | 5,47% | 0,10% | Nhóm thực sự chọn |
| `bottleneck2__fast2_slow634` | 14,16% | 9,86% | 0,10% | Nhóm thực sự chọn |
| `bottleneck2__fast2_slow436` | 12,08% | 6,02% | 0,10% | Nhóm thực sự chọn |
| `overlap3__fast1_slow325` | 8,33% | 3,68% | 3,42% | Tranh chấp; cơ hội có đổi cost |
| `overlap3__fast1_slow352` | 8,63% | 3,68% | 3,39% | Tranh chấp; cơ hội có đổi cost |

38/48 ứng viên trượt gate tranh chấp; 45/48 trượt cơ hội có đổi cost; 45/48 trượt nhóm thực sự chọn. **48/48 đạt availability ≥25% và pending cuối đoạn đo ≤1**. Các tập trượt chồng lấp, không cộng các số này thành tổng ứng viên.

`NO_ELIGIBLE_SETTING` là kết quả screening theo ngưỡng, không phải kiểm định rằng mọi setting đều vô dụng. Không có CI xác nhận winner vì không có winner. Năm đối chứng trên panel mới đều có hai rate cost-sensitive bằng 0; uniform vẫn có strict contention 11,17% nhưng không có chênh lệch cost khi đổi người. Cancellation có duration khác nhau nhưng tổng duration khi đổi người vẫn bằng nhau, cũng được loại đúng.

## Vì sao bottleneck2 nhìn mạnh nhưng ít có phân công để học?

Ví dụ `bottleneck2__fast2_slow634` có thời gian sửa thực tế:

| Máy | Technician A | Technician B |
|---|---:|---:|
| M0 | 2 | Không đủ skill |
| M1 | 2 | Không đủ skill |
| M2 | 2 | 3 |
| M3 | 2 | 4 |

Chữ số 6 trong tên pattern thuộc hàng bị mask, **không phải thời gian sửa trên một cạnh dùng được**. M2 và M3 có thể đổi người: tổng thời gian là 5 hoặc 6 tick, chênh 6 đơn vị chi phí unavailable. Nhưng M0/M1 chỉ A sửa được, và rule thường ưu tiên một trong các máy này; khi đó A đã bị cố định, B sửa máy chung còn lại, không có cách đổi người trong cùng nhóm đã chọn.

Ở matching: 312/3.072 tick có cơ hội đổi cost, chỉ 3 tick chọn được nhóm đó; **309/309 tick bỏ qua cơ hội đều chọn đủ cardinality nhưng chứa máy chỉ một người sửa được**. Greedy tương ứng 303 cơ hội, 3 lần chọn và 300/300 lần bỏ qua có máy độc quyền. Đây không phải rule chọn thiếu máy. Khó khăn chủ yếu nằm ở chọn nhóm máy và phân bổ công suất, ít nằm ở đổi technician trong nhóm đã chọn.

## Overlap3 có hậu quả phân công thật, và metric phụ thuộc policy

`overlap3__fast1_slow325` là ứng viên đáng nghiên cứu tiếp, chưa phải winner:

| Máy | Technician A | Technician B |
|---|---:|---:|
| M0 | 1 | Không đủ skill |
| M1 | 1 | 3 |
| M2 | 1 | 2 |
| M3 | 1 | 5 |

Mức phục hồi trên mọi cạnh hợp lệ là 0,75. Ví dụ cùng sửa M2 và M3: A→M3, B→M2 cần tổng 3 tick; A→M2, B→M3 cần 6 tick. Chênh lệch là 18 đơn vị chi phí unavailable. Cả hai cách đều hợp lệ, sửa đúng cùng hai máy, phục hồi cùng mức. Đây là khác biệt dịch vụ thật, không còn chỉ đổi tên hai máy tương đương.

| Rule, 64 seed screening | Tranh chấp | Có cơ hội đổi cost | Nhóm thực sự chọn có đổi cost | Cost toàn episode + tail |
|---|---:|---:|---:|---:|
| deadline_matching | 8,33% | 3,68% | 3,42% | 466,42 |
| priority_greedy | 10,84% | 7,65% | 7,39% | 507,75 |

Greedy tự nó đạt cả năm ngưỡng điểm screening; matching làm hai rate đầu xuống dưới ngưỡng, nên phép lấy worst-rule loại setting này. **Không nên diễn giải rate thấp hơn dưới matching thành bằng chứng môi trường không có bài toán phân công:** hành động trước đó ảnh hưởng lúc máy hỏng, tuổi máy và lúc technician rảnh, nên rate này phụ thuộc trajectory do policy tạo ra.

Chẩn đoán hậu nghiệm trên đúng các episode đã ghi:

- Matching chọn 105 tick tranh chấp có cùng nhóm máy đổi cost được; cả 105 đều dùng assignment có chi phí cam kết thấp nhất trong nhóm đó.
- Greedy chọn 227 tick như vậy; **205/227** có assignment khác trên đúng cùng nhóm máy với chi phí cam kết thấp hơn. Tổng chênh lệch thành phần cam kết là 3.138 đơn vị, không được coi là tổng tiết kiệm toàn episode nếu thay policy.
- Cost matching thấp hơn greedy 41,33 đơn vị, khoảng **8,14%**. Đây là so sánh điểm khám phá trên dữ liệu screening của ứng viên xem hậu nghiệm, không có xác nhận độc lập; nó gộp chọn máy, phân công và động lực tương lai. Không thể quy toàn bộ khoảng cách cho assignment hoặc coi là mức lợi ích của RL.

Cost cam kết ở đây là `6 × tổng thời gian sửa của các job mới`. Chi phí khởi động PM/CM triệt tiêu khi giữ cùng nhóm máy; các job này hoàn tất trong horizon + tail. Nhưng hỏng máy tương lai, mức chờ và các job khác có thể bù lại lợi ích thành phần này. Matching tối thiểu hóa thành phần hiện tại cũng chưa chứng minh tối ưu return dài hạn.

## Hướng tiếp theo

Giữ nguyên kết luận v3 và các gate của nó: chưa chốt benchmark được xác nhận. Không hạ ngưỡng sau khi xem kết quả, không chuyển một ứng viên screening thành winner.

Trước khi quét tiếp, cần phân biệt mục tiêu **môi trường có cơ hội phân công mang giá trị** với mục tiêu **cả hai rule đều thường xuyên đi vào các trạng thái tranh chấp**. V3 cho thấy hai mục tiêu này có thể khác nhau. `overlap3__fast1_slow325` có bằng chứng cụ thể cho mục tiêu thứ nhất, trong khi không đạt protocol của mục tiêu thứ hai.

Đề xuất cho một protocol mới: ưu tiên xác nhận ứng viên này trên seed mới chưa sử dụng, đo tần suất và độ lớn chênh lệch assignment cùng nhóm máy, và so sánh return khi can thiệp assignment để tách nó khỏi chọn máy. Đây chỉ là đề xuất, chưa triển khai hoặc chạy. Phép so sánh này phục vụ đánh giá; RL vẫn học chọn máy, technician và WAIT từ reward, không cần nhãn từ rollout tương lai. Chưa có dữ liệu chứng minh RL sẽ học được hay vượt matching.

## Fallacy scan — 11/11 đã kiểm tra

| Loại | Kết quả / cách giới hạn diễn giải |
|---|---|
| Simpson | Báo cáo riêng hai rule; không dùng trung bình che rule không đạt. |
| Ecological | Rate theo seed trajectory không suy thành năng lực từng technician hay mọi policy. |
| Berkson | Các ứng viên nêu chi tiết được xem hậu nghiệm; không suy ra toàn không gian setting. |
| Collider | Chẩn đoán trên nhóm thực sự chọn là điều kiện theo policy, không kết luận nhân quả return. |
| Base rate | Mẫu số 48 tick được nêu; không lẫn rate toàn lượt với rate điều kiện có cơ hội. |
| Regression to mean | Ứng viên nổi bật chưa được xác nhận trên seed mới; không đổi vai panel controls. |
| Survivorship | Đủ toàn bộ grid và episode; báo cáo cả 48 ứng viên không đạt. |
| Look elsewhere | Có 48 ứng viên; ví dụ và cost gap là khám phá, không kiểm định xác nhận. |
| Forking paths | Giữ gate/ranking khóa trước run; diagnostic hậu nghiệm được đánh dấu riêng. |
| Correlation/causation | Cost gap hai rule không tách riêng assignment; local obligation không thành RL gain. |
| Reverse causality | Policy vừa tác động trajectory vừa tác động rate; rate thấp có thể đi cùng hành động tốt. |

## Tái lập phân tích

`analyze.py` cùng thư mục là script chỉ đọc artifact, dùng source checkout gắn commit trên. Kết quả có trong `analysis.json`, log trong `validation.log`. Script thực hiện replay để đối chiếu các episode đã có; không screening setting mới, không huấn luyện và không mở thêm outcome ngoài run đã tải về.
