# Maintenance coupling v1 — phân tích full run

## Material Passport
- Origin Skill: academic-research-suite / experiment-agent, validate.
- Origin Date: 2026-10-04; Verification Status: ANALYZED, có arithmetic/decision replay consistency checks; không coi là tái tạo độc lập simulator.
- Source: `lab_results/maintenance_coupling_full_cpu_20261004T133132Z/`.
- Source commit: `07697c29a6e563b762ef6497b5949366ae85c554`; protocol `maintenance_coupling_v1`.
- Derived artifacts: thư mục chứa báo cáo này; raw files giữ nguyên.

## Kết luận chính

**Primary gate PASS và stronger mechanism gate PASS theo protocol đã khóa.** Khi nguồn lực hạn chế và thời gian sửa khác theo kỹ thuật viên, joint timing–assignment lookahead C có lợi ích vượt baseline B; control D cho thấy việc được thay đổi thời điểm bảo trì có đóng góp ngoài cost-aware assignment lookahead. Tín hiệu giới hạn ở một số cấu hình tranh cùng chuyên môn, không xuất hiện đồng đều trên toàn panel.

Problem cụ thể được quan sát: **các nhu cầu bảo trì có thể tự hội tụ vào cùng thời điểm, cùng một thợ chuyên môn; chủ động điều chỉnh thời điểm yêu cầu bảo trì có thể giãn nhu cầu đó và tránh rủi ro chờ.** Nhu cầu không hoàn toàn là danh sách task cố định để giải assignment: policy có thể tác động thời điểm nhu cầu phát sinh.

## Tách nguồn và verification

Run coupling có manifest COMPLETED, source clean, CPU/Python 3.12.3; chạy từ 13:31:32 đến 13:31:49 UTC. Source/protocol SHA256 khớp nội dung Git tại commit nguồn. Đủ 7.680 unique episode rows, 288 development rows, 2.304 decision rows, 32 exact-oracle rows và 48 coordination rows. `episodes.partial.csv` khớp byte với final `episodes.csv`; chỉ tính final một lần.

Danh tính và seed keys khớp đầy đủ Cartesian product đã khóa. Dev/evaluation không chồng seeds; cấu hình realized khớp generator; thời gian sửa trung bình theo hàng được giữ bằng nhau qua bốn cells. Mọi episode có zero invalid requests, chi phí hữu hạn và khớp tổng components + terminal liability.

Đã replay **7.680 episodes và 2.304 quyết định**, đối chiếu action/state/next state và mọi numerical endpoint trừ wall-clock timing; sai số tối đa bằng 0. Tính lại 32 exact policy/oracle rows, khớp raw. Đây là consistency audit dùng cùng pinned implementation; không chứng minh độc lập rằng toàn bộ simulator/oracle không còn lỗi. Independent tiny oracle/tree tests đã có ở giai đoạn verification trước push.

Các file tải kèm từ `ra_qmix_heuristic_assignment_full_cpu_20261004T124714Z` có manifest local RUNNING; không đưa vào phân tích. Không suy từ snapshot local rằng trạng thái remote hiện tại vẫn đang chạy. `ra_qmix_td_assignment_full_cpu_20261004T053816Z` là run khác dù COMPLETED, cũng loại khỏi bảng. Không so ngang reward/cost của simulator cũ với objective mới.

Threshold offset chọn trên development là 1. Mean B cost của offsets 0/1/2 lần lượt 49,625 / 22,375 / 30,4583; policy.json khớp argmin. C depth=3, tail=2; D giữ trigger A/B và dùng cùng depth, tail, expected-cost objective với C.

## Bảng cost chính

Cost thấp hơn tốt hơn; mỗi số là mean của 12 instance means, mỗi instance có 40 shock seeds.

| Điều kiện | A: health + fastest | B: health + queue assignment | C: joint lookahead | D: fixed-timing lookahead |
|---|---:|---:|---:|---:|
| 4 thợ, đồng đều | 33,546 | 20,917 | 20,917 | 20,917 |
| 4 thợ, khác chuyên môn | 23,292 | 19,083 | 19,250 | 19,250 |
| 2 thợ, đồng đều | 33,546 | 20,917 | 20,917 | 20,917 |
| 2 thợ, khác chuyên môn | 23,292 | 23,292 | **19,417** | 21,171 |

Primary cell: B−C = **3,875**, percentile paired-bootstrap CI95% **[1,225; 6,883]**. Mean per-instance relative savings = **11,9531%**, vượt minimum practical effect 5%; CI absolute lower >0. Ratio từ hai grand means là 16,6369%, khác primary 11,9531% do cách lấy trung bình; không thay endpoint bằng ratio hấp dẫn hơn.

Control D−C tại primary cell = **1,7542**, CI95% **[0,1667; 3,7542]**. Điều này hỗ trợ lợi ích của timing flexibility trong recipe/model đang xét. Không suy mọi thuật toán tách rời đều kém hoặc joint formulation là điều kiện cần về lý thuyết.

Interaction `(B−C_high,heterogeneous − B−C_high,uniform) − (B−C_low,heterogeneous − B−C_low,uniform)` = **4,0417**, CI95% **[1,225; 7,3583]**. Gate mạnh hơn đạt đồng thời primary, interaction và D−C. CI được tính lại bằng bootstrap instance-level riêng, khớp summary; n=12, không phải n=7.680.

Cell low/heterogeneous: B−C = −0,1667, CI95% [−0,6667; 0,3333]; C hơi tốn hơn theo point estimate. Hai uniform cells có B−C = 0 trong panel này. Không kết luận C thắng ở mọi điều kiện.

## Cơ chế và độ tập trung

Trong high/heterogeneous, C thắng B ở **5/12 cấu hình**, hòa 7, không có cấu hình mean cost xấu hơn. C thắng D ở **4/12**, hòa 8. Hiệu ứng timing vì vậy có tính điều kiện, không phải cải thiện chung ở mọi seed.

Có 8 cấu hình hai máy cùng ưu tiên một thợ, 4 cấu hình chuyên môn bổ sung nhau. Toàn bộ 5 cấu hình có B−C >0 nằm trong nhóm cùng ưu tiên một thợ. Đây là breakdown exploratory sau khi đọc outcomes, không thay primary protocol và không là xác nhận thống kê riêng cho subgroup. Trong các trường hợp cùng thợ nhưng initial ages đã giãn đủ, C có thể hòa B.

“High pressure” có 2 thợ cho 2 máy, nên vẫn có thể phục vụ hai máy đồng thời bằng hai người. Scarcity có ý nghĩa ở nhóm chung chuyên môn là chỉ có **một thợ nhanh** cho cả hai máy; low pressure bổ sung một bản sao của chuyên môn đó. Kết quả nói trực tiếp hơn về thiếu năng lực chuyên môn phù hợp, không chứng minh thiếu tổng headcount trong mọi hệ thống.

Witness từ trace đã lưu: instance 780200, high/heterogeneous, initial ages [1,1], failure age 5, cả hai service rows [2,4]. Threshold của A/B/D là age 4:

- B đợi tới step 3, khi hai máy đều age 4, rồi yêu cầu [1,1]. Một máy được sửa, máy kia chờ và đạt age 5 — chịu rủi ro hỏng trong lúc chờ. Mean cost qua 40 shocks là 26,7. Episode trace đầu tiên không hỏng thực tế; witness cho thấy phơi nhiễm rủi ro, không phải khẳng định có hỏng trong trace đó.
- D cũng đợi tới step 3, nhưng chọn [1,2]: tránh chờ bằng cách dùng một thợ mất 4 bước. Mean cost 18.
- C yêu cầu sửa một máy ngay step 1 (age 2), rồi sửa máy còn lại tại step 3: hai việc dùng thợ nhanh được giãn theo thời gian. Mean cost 16.

Lưu ý B tối ưu tổng completion times, không expected failure cost. Trong witness này [1,1] và [1,2] đồng hạng completion-time và lexicographic tie chọn [1,1]. Một phần B−C có thể là hạn chế objective/tie rule của baseline. D được thêm trước sealed outcomes để giảm nhầm lẫn đó; chênh lệch C−D vẫn có nhưng nhỏ hơn C−B.

## Tổn thất giảm từ đâu, và cái giá phải trả?

Mean high/heterogeneous qua cùng shock panel:

| Metric / episode | B | C | D |
|---|---:|---:|---:|
| Failures | 0,2063 | **0** | 0,0875 |
| Queue machine-steps | 0,5833 | **0** | 0,1667 |
| Planned down machine-steps | 7,8333 | **9,0833** | 8,9167 |
| Failed down machine-steps | 0,0458 | 0 | 0 |
| Maintenance count | 5,4167 | 5,0000 | 4,8333 |
| Stage cost | 22,8750 | 19,0833 | 20,5042 |
| Terminal liability | 0,4167 | 0,3333 | 0,6667 |

B−C cost decomposition: maintenance +0,8333; planned down −1,2500; failure +3,5500; failed down +0,3667; queue +0,2917; terminal +0,0833 = **3,8750** savings. Vì vậy C giảm failure/queue costs nhưng tăng planned downtime. Khoảng 97,85% realized saving nằm ở stage cost, không phải chênh lệch terminal accounting; điều đó không loại ảnh hưởng của horizon/closeout lên quyết định trước đó.

Zero C failures trong 480 rollouts của primary cell không tự chứng minh reliability tuyệt đối ngoài model/panel. Simulator dừng aging khi sửa; repair kéo dài có thể giảm exposure vận hành và số lần bảo trì. Với planned downtime cost chỉ 1 so với failure cost 12–24, trade-off này có thể được ưu tiên. **Không diễn giải cost giảm thành production/availability tăng.** Tổng planned+failed down machine-steps của C còn cao hơn B trong primary cell.

## Oracle và computation

C đạt exact optimal expected cost ở **8/8 tiny cases**, horizon 4. B đạt 6/8, D đạt 7/8. A không đạt optimum trong 8 cases đó. Không suy C tối ưu cho horizon 12 hoặc quy mô lớn; không ghép tiny oracle denominator vào full rollout cost.

Mean measured C decision time là khoảng 2,6 ms/episode ở primary cell so với B khoảng 0,06 ms/episode; cache được tái sử dụng giữa episodes. Con số này là amortized trên benchmark nhỏ, không phải worst-case deployment latency hay phép so throughput dưới hardware control. C nhanh đủ cho diagnostic này, chưa đánh giá scaling.

## Ý nghĩa cho hướng nghiên cứu

Đã có bằng chứng cho một vấn đề dễ hiểu: **điều khiển thời điểm để tránh các yêu cầu bảo trì cùng tranh một chuyên môn**. Control D cho thấy bài toán không chỉ là chọn người nhanh/chọn người đang rảnh ở thời điểm trigger. Một baseline chủ động giãn lịch theo health slack và lịch thợ là bước so sánh hợp lý tiếp theo; kết quả hiện tại chưa yêu cầu phải dùng MARL/QMIX.

Các follow-up về nhiều máy hơn, horizon dài hơn, sensitivity của planned-downtime cost và observation/RUL uncertainty cần protocol mới/fresh configs. Đặc biệt cần kiểm tra policy có giữ production availability hợp lý hay chỉ mua reliability bằng thời gian ngừng máy nhiều hơn. Không thêm module dựa riêng trên việc C thắng B ở benchmark này.

## Fallacy scan và giới hạn

11/11 kiểu được rà soát:

| Kiểu | Đánh giá |
|---|---|
| Simpson’s paradox | Báo riêng bốn cells; không che cell C hơi tốn hơn. |
| Ecological fallacy | Báo 5/12 và 4/12 wins; mean không đại diện mọi cấu hình. |
| Berkson’s paradox | Không lọc bỏ cấu hình hòa; subgroup cùng chuyên môn được ghi exploratory. |
| Collider bias | Primary không condition vào realized failures/waiting. |
| Base rate neglect | Báo số failure thực tế và control cells; zero failures có phạm vi. |
| Regression to mean | Frozen fresh panel; không chọn worst prior seeds rồi báo recovery. |
| Survivorship bias | Đủ Cartesian product; incomplete experiment khác không dùng. |
| Look-elsewhere effect | Primary/stronger gates khóa trước full; subgroup/trace secondary. |
| Garden of forking paths | Không đổi offset/depth/gate sau evaluation; phân biệt relative estimator. |
| Correlation vs causation | Controlled policy intervention hỗ trợ effect trong simulator, không chứng minh mechanism duy nhất hay necessity/novelty. |
| Reverse causality | Policies đi trước outcomes; cost decomposition không tự là causal mediation. |

Bootstrap percentile CI với 12 instance draws có thể nhạy panel composition. Model chính xác và fully observed, hai máy, horizon 12, không job-shop/production revenue, không RUL noise, specialization chỉ service duration. A có tie rule rất đơn giản nên A improvements không là bằng chứng mạnh nhất; B và D là đối chiếu chính. Replay dùng cùng implementation nên không là độc lập xác nhận model validity. Giữ nguyên raw results và cả những limitations này khi viết proposal/paper.

Inference còn có điều kiện trên panel 40 shock seeds dùng chung cho các instances: bootstrap resamples instances, không resample shock panel. Vì vậy CI không bao gồm đầy đủ độ bất định khi thay một bộ shock seeds hoàn toàn mới; không diễn giải nó thành độ tin cậy vô điều kiện cho mọi deployment.
