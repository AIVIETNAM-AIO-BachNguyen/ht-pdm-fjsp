## Material Passport

- Workflow: ARS experiment-agent / validate, xử lý trực tiếp trong phiên.
- Verification Status: ANALYZED. Tính lại thống kê từ CSV; replay nhất quán cùng implementation, không phải tái lập độc lập.
- Nguồn: `/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/maintenance_stagger_full_cpu_20261004T150819Z`.
- Protocol/source khóa: `maintenance_stagger_v1`, commit `ccaff3e1e36078b9258cbb209089cce6b5ca684c`.
- Phạm vi: ba policy B/E/C; 16 instance × 30 shock × 24 cell. Không trộn các experiment khác.

## Kết luận

E là baseline đơn giản đáng giữ: giảm chi phí so với B trong cell chính và đạt biên không kém C. Nhưng gate tổng hợp FAIL: C không chứng minh được lợi thế trước B ở horizon 24; E không chứng minh được rằng mất thời gian vận hành tối đa 2 điểm phần trăm. Kết quả không ủng hộ kết luận rằng joint lookahead hiện tại là giải pháp tốt hơn về vận hành.

B = ngưỡng sức khỏe + phân công tối thiểu tổng thời gian hoàn thành; E = lịch đặt chỗ theo hạn an toàn với chi phí/rủi ro; C = tối ưu kỳ vọng nhìn trước 3 bước, cộng 2 bước baseline. C không phải oracle toàn horizon.

## Cell chính đã khóa: high / heterogeneous / price 1 / horizon 24

| Policy | Tổng chi phí | Chi phí/bước | Operating proxy | Hỏng/episode | Queue machine-steps | PM machine-steps |
|---|---:|---:|---:|---:|---:|---:|
| B | 48.4729 | 2.0197 | 60.71% | 0.2042 | 0.5625 | 18.8125 |
| E | 44.6250 | 1.8594 | 58.98% | 0.0000 | 0.0000 | 19.6875 |
| C | 45.8167 | 1.9090 | 51.71% | 0.1479 | 0.0000 | 23.1792 |

Operating proxy = 1 − (planned + failed machine-steps)/(2 × horizon). Đây là tỷ lệ thời gian hoạt động trong simulator, không phải throughput, makespan hay doanh thu sản xuất.

| Gate | Kết quả từ paired instance bootstrap | Phán định |
|---|---|---|
| Comparator valid: B−C CI dưới > 0 | 2,6563; CI95 [−0,7750; 6,1875] | FAIL: chưa chứng minh C tốt hơn B |
| E không kém C: CI trên của (E−C)/C ≤ 5% | −1,91%; CI95 [−6,35%; +1,36%] | PASS: không kém theo biên đã khóa; chưa chứng minh hơn C |
| E hữu ích: B−E CI dưới > 0 và giảm tương đối trung bình ≥ 5% | 3,8479; CI95 [1,2250; 6,9167]; giảm 6,80% | PASS |
| Bảo vệ vận hành: CI dưới của E−B ≥ −2 pp | −1,7274 pp; CI95 [−2,9644; −0,6858] pp | FAIL: chưa loại trừ mất hơn 2 pp |
| Joint operational | Hai điều kiện không đạt | FAIL |

Giảm tương đối 6,80% là trung bình (Bᵢ−Eᵢ)/Bᵢ qua 16 instance, CI95 [2,34%; 11,79%]. Tỷ số từ hai grand means là 7,94%, một endpoint khác. E thắng B 7/16 instance, hòa 9/16, không thua; toàn bộ 7 thắng đều thuộc dạng hai máy cùng thích một thợ. Có 10 instance shared preference, 6 complementary preference; không suy diễn rằng mọi shared instance đều cải thiện. C−B operating proxy = −9,0017 pp, CI95 [−12,8255; −5,2257].

## Độ nhạy giá downtime và horizon

Các cell sau là phân tích thứ cấp mô tả, không điều chỉnh family-wise multiplicity. Không dùng cell h12 để thay cell chính h24 sau khi thấy kết quả.

| Giá downtime | Horizon | B cost/bước | E cost/bước | C cost/bước | E giảm so với B, mean instance-relative |
|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 2.1540 | 1.8333 | 1.8396 | 11.37% |
| 1 | 24 | 2.0197 | 1.8594 | 1.9090 | 6.80% |
| 4 | 12 | 4.4821 | 4.3438 | 5.1337 | 2.36% |
| 4 | 24 | 4.4181 | 4.3646 | 5.6592 | 1.14% |
| 8 | 12 | 7.5863 | 7.5347 | 7.8944 | 0.55% |
| 8 | 24 | 7.6161 | 7.5903 | 8.7444 | 0.32% |

Ở h24, tăng giá downtime 1→4→8 làm tiết kiệm tương đối E/B giảm 6,80%→1,14%→0,32%. Các CI lợi ích E/B ở giá 4/8 bao gồm 0. E qua operating guard ở giá 8 nhưng lợi ích chi phí không còn được xác lập. C ở h24 có tổng chi phí cao hơn B 29,7854 tại giá 4 (CI95 [2,7750; 63,5708]) và 27,0813 tại giá 8 (CI95 [8,3792; 47,7813]); đây là kết quả sensitivity, không một kiểm định confirmatory mới.

Trong 18/24 cell còn lại (low capacity pressure hoặc uniform skill), E và B bằng nhau về tổng chi phí và operating proxy. Lợi ích E/B hiện tập trung vào high + heterogeneous. Tại price 1, mức giảm chi phí tuyệt đối trung bình vẫn 3,8479 ở cả h12 và h24: chưa có bằng chứng lợi ích tích lũy tỷ lệ với thời gian. Hai instance thay đổi −4/+4 nên không nói mọi instance giữ nguyên. Đây là dấu hiệu cần kiểm tra transient/initial synchrony, không chứng minh chỉ có tác dụng ở khởi tạo.

## Cơ chế từ trace đã replay

Dev chọn B offset = 1 ở cả ba giá; E lead = 0, 1, 0 lần lượt giá 1, 4, 8. Với lead=0, E không khởi động trước due: không thể quy lợi ích ở giá 1/8 cho chủ động bảo trì sớm. E vẫn có thể thay đổi thời điểm khi không có slot, nhưng cơ chế trace sau là thay phân công để tránh queue.

Witness khóa theo raw: instance 860210, shock 861200, high/heterogeneous/price1/h24. Hai máy cùng service [2,4], ages ban đầu [1,1], failure_age=3, p=0,4. Ở step1 cả hai ages=2: B action [1,1] đưa hai máy vào thợ nhanh; E action [1,2] dùng cả thợ nhanh và chậm. Máy thứ hai của B hỏng ở step2 trong trace này; E không có hỏng. Trong 30 shock, mean cost B=66,2, E=50,0; E mất 3,125 pp operating proxy trên instance này. Điều này minh họa đổi downtime lấy giảm failure risk.

Tại step1, assignment B [1,1] có tổng completion = 2+4=6; [1,2] cũng 2+4=6. Tie-break lexical chọn queue trên thợ nhanh. Do đó một phần khoảng cách xuất phát từ mục tiêu/tie-break của B không xét hazard, chứ chưa chứng minh cần một kiến trúc học phức tạp. Không thể xác lập mọi lợi ích chỉ do tie-break nếu chưa có ablation khóa riêng.

Phân rã mean cost E−B trong cell chính:
- Maintenance: -0.375000.
- Planned downtime: +0.875000.
- Failed downtime: -0.366667.
- Failure penalty: -3.762500.
- Queue: -0.281250.
- Terminal closeout: +0.062500.

Tổng −3,847917. E loại bỏ failures và queue trên panel này, nhưng planned_steps tăng 18,8125→19,6875. Không được đồng nhất giảm cost với tăng thời gian sản xuất.

Witness C không bền: instance860209/price1/h24, mean cost C=66,0667 so với B=52,0667 và E=46. Trong shock861200, C trì hoãn máy đến hỏng ở steps4,9,14; sau đó sửa. Trace cho thấy failure-risk thực tế còn tồn tại dù C tính kỳ vọng. Đây không phải bằng chứng nhìn trước luôn kém; chỉ là phản ví dụ cho comparator đang dùng.

## Giới hạn mô hình và điều cần kiểm tra tiếp

Source cho máy ngừng già hóa trong suốt service, không có production workload/revenue, và operating proxy chỉ tính trong horizon còn cost có terminal closeout. C đánh giá 3 bước tối ưu + 2 bước baseline rồi closeout tại chân trời cục bộ: tuổi khỏe của máy ở chân trời đó không có giá trị tiếp diễn ngoài phần closeout outstanding work. Hai đặc điểm này có thể tạo động cơ chọn thợ chậm hoặc trì hoãn rủi ro do tầm nhìn ngắn. Trace xác nhận có chọn thợ chậm và trì hoãn; nguyên nhân terminal/value chưa được tách bằng experiment nên ghi là giả thuyết cơ chế.

Hướng kiểm chứng tiếp phù hợp là bài toán đổi failure risk lấy mất công suất, với workload sản xuất/đo throughput rõ ràng và nguồn tranh chấp lặp lại. Trước khi thiết kế thuật toán mới, cần xác minh comparator cost/risk-aware đơn giản và kiểm tra C với continuation hợp lý. Không đổi dữ liệu hoặc mở seed mới trong lần phân tích này.

## Audit và tái lập

- Completed, source sạch và đúng commit; SHA256 source/dependency/protocol khớp Git.
- Đủ đúng 34.560 episodes, 7.680 development rows, 384 resolved config và coordination rows, 20.736 decision rows; không thiếu/duplicate Cartesian keys.
- episodes.csv trùng episodes.partial.csv; cost decomposition, cost/bước, operating proxy, finite values và invalid action = 0 đều hợp lệ.
- Dev/eval seed tách rời; tham số đã chọn đúng tối thiểu dev cost/bước; không dùng eval để chọn lại.
- Tính lại tất cả endpoint/CI và năm gate từ CSV bằng aggregation/bootstrap arithmetic riêng; khớp summary.
- Replay 1.152 episode (shock đầu mỗi cell) và toàn bộ 20.736 decision/state/transition bằng source snapshot khóa: sai số 0; runtime không so sánh. Không replay 33.408 shock-episodes còn lại. Đây là consistency audit cùng implementation, không independent reproduction.
- Runtime cùng CPU/run được lưu trong statistics; cache C chia sẻ giữa episode, execution order cố định. Không dùng tỷ số runtime này để tuyên bố hiệu suất tổng quát.
- Inference unit = 16 instance means, CI conditional trên chung panel 30 shocks. 34.560 episode không phải cỡ mẫu độc lập. Bootstrap percentile khóa seed862100, 10.000 resamples; CI [draw250,draw9750].

## Kiểm tra sai lầm thống kê: 11/11

| Loại | Đánh giá trong phân tích này |
|---|---|
| Simpson | Giữ 24 strata; không gộp giá/horizon/skills thành một hiệu ứng. |
| Ecological | Suy luận tại instance; không dùng số episode để phóng đại n. |
| Berkson | Không lọc theo kết quả; generator chỉ N=2 nên external validity giới hạn. |
| Collider | Không điều chỉnh theo hỏng/downtime phát sinh; chỉ dùng chúng giải thích trace. |
| Base rate | Báo cả mean failures, downtime, costs; không chỉ % savings. |
| Regression to mean | Panel mới tách seed, paired comparators; không chọn lại instance thắng cũ. |
| Survivorship | Đủ mọi Cartesian key; không loại run/episode xấu trong panel đã khóa. |
| Look-elsewhere | Cell chính/gate giữ nguyên; các sensitivity/subgroup mô tả, không claim confirmatory đã hiệu chỉnh. |
| Forking paths | Protocol/commit khóa; không retune trên eval, không thay primary sang h12. |
| Correlation vs causation | So sánh policy can thiệp trong simulator; không suy ra hiệu quả xưởng thực tế. |
| Reverse causality | Action trước transition; giới hạn là objective/model validity, không đảo trình tự. |

`analyze.py`, `audit.json`, `statistics.json`, `instances.csv`, source snapshots và protocol snapshot nằm cùng thư mục báo cáo.
