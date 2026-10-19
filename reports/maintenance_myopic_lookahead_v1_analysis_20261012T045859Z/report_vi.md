## Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent, inline.
- Origin Mode: analysis and read-only reconstruction of downloaded artifacts.
- Origin Date: 2026-10-12.
- Verification Status: **ANALYZED** cho diễn giải; **VERIFIED** cho source binding, toàn bộ replay, oracle metric và tái dựng thống kê dưới đây.
- Version Label: maintenance_myopic_lookahead_v1.

# Joint lookahead có lợi ích; assignment lookahead gặp bẫy cửa sổ ngắn

**Joint lookahead giảm cost trung bình 3,14%, nhưng chưa đạt gate xác nhận cải thiện ít nhất 1%: cận dưới CI 99% chỉ 0,84%.** Đối chứng chỉ đổi technician làm cost tăng 1,05%, với CI 99% cùng chiều bất lợi. Có bằng chứng lợi ích của joint policy so với myopic; chưa có bằng chứng assignment lookahead hiện tại cải thiện assignment.

Giữ `overlap3__fast1_slow325` là ứng viên khám phá. Run này không biến nó thành winner của v3, chưa kiểm tra RL và không phải một lần quét setting mới.

## Tính toàn vẹn và phạm vi

Run `maintenance_myopic_lookahead_v1_full_cpu_20261012T045644Z` hoàn tất CPU Linux trong **76,16 giây**, đúng commit sạch `ac9fc61316d26a75fc3a5869e084d4476bf828db`. Đủ **128 seed mới × 3 method =384 episode**, **23.040 transition vật lý**, **896 landmark chung** và **2.366 root row**. Mỗi episode đo H48, sau đó L12 continuation myopic chung, không reset.

Kiểm tra trên Mac:

- 21 SHA256/size artifact và source seal khớp; tám module, ba registry kế thừa, protocol và registry mới khớp commit. Payload 33.144.814 byte.
- Episode CSV hoàn chỉnh bằng byte với CSV streaming cuối cùng; đúng grid/count, seed audit PASS, không invalid assignment.
- Mở rộng phạm vi audit từ hai seed của runner lên **toàn bộ 128 seed**, giữ nguyên tham số algorithm. Replay mọi action, state, event, cost, metric và restrictions; tái dựng mọi lựa chọn planner và mọi landmark trên đúng seed dự báo đã ghi. Tái dựng **42.316.800 forecast transition** khớp; không chạy policy mới hay mở panel mới.
- Oracle độc lập dùng đệ quy theo worker kiểm tra matching/cardinality, strict contention, physical choices và các span cost trên **18.432 trạng thái đo**; mọi metric khớp.
- Bootstrap độc lập bằng count weights/einsum, thay cách indexed resampling của runner: cả năm CI 99%, mean của ba method, các gate và hai CI cơ chế 95% đều khớp.
- Hash toàn bộ artifact được kiểm tra lại sau phân tích. Raw data, protocol và experiment code không bị sửa.

Setting cố định: thời gian sửa M0..M3/T0,T1 là `[(1,0),(1,3),(1,2),(1,5)]`, mọi cạnh hợp lệ phục hồi .75. Lookahead nhìn 12 tick/128 scenario, sau root dùng myopic continuation trong forecast; trên trajectory thật nó replan mỗi tick H. Forecast không đọc future shock thật. Myopic là score urgency + restoration/duration ở v3, không phải bộ tối ưu economic cost một bước.

## Kết quả chính trên panel mới

| Method | Cost H+L | Cost H | Cost L | Availability H | Pending H / tail |
|---|---:|---:|---:|---:|---:|
| Myopic matching | 467,77 | 376,07 | 91,70 | 71,65% | 0 / 0 |
| Joint lookahead | 453,09 | 365,13 | 87,95 | 72,73% | 0 / 0 |
| Assignment lookahead | 472,66 | 379,92 | 92,74 | 71,28% | 0 / 0 |

Relative gain dùng tỷ số mean cost, bootstrap theo seed paired, không dùng trung bình phần trăm từng episode. Năm interval được khóa ở 99%, Bonferroni năm với mức family coverage danh nghĩa ít nhất 95%; percentile bootstrap vẫn là xấp xỉ.

| So sánh / guard | Estimate | CI 99% | Diễn giải |
|---|---:|---:|---|
| Joint giảm cost so với myopic | **3,14%** | **[0,84%; 5,40%]** | CI trên 0, nhưng cận dưới chưa ≥1% |
| Assignment giảm cost so với myopic | **−1,05%** | **[−1,93%; −0,33%]** | Assignment lookahead tốn hơn |
| Joint giảm cost so với assignment | 4,14% | [2,20%; 6,19%] | Joint tốt hơn ablation này |
| Availability joint − myopic | +1,074 điểm % | [+0,305; +1,872] điểm % | PASS guard ≥−1 điểm % |
| Pending joint − myopic | 0 | [0; 0] | PASS guard ≤0,10 máy |

**Scientific status `NOT_CONFIRMED` chỉ vì practical-cost gate.** Không diễn giải thành “joint không có tác dụng”: CI primary nằm hoàn toàn phía cải thiện, chỉ chưa đủ đảm bảo mức cải thiện tối thiểu đã khóa. Không hạ margin, đổi confidence hoặc thêm seed sau khi xem kết quả để ép gate qua. Pending/overdue đều bằng 0 trong mẫu đã quan sát; bootstrap [0,0] phản ánh mẫu này, không chứng minh mọi trajectory tương lai đều không có backlog.

## Lợi ích nằm ở đâu?

Joint tiết kiệm trung bình **14,69 đơn vị/episode**. Thành phần unavailable giảm 16,36; maintenance tăng 0,85 và failure cost tăng 0,82; waiting không đổi. Lợi ích có ở cả H (10,94) và common tail (3,75), nên không chỉ là thay đổi cost tại cutoff H. Số failure trong H vẫn tăng từ 0,227 lên 0,281/episode; giảm total cost không đồng nghĩa mọi thành phần đều giảm.

Assignment tốn thêm **4,89 đơn vị/episode**, chủ yếu unavailable tăng 5,30; maintenance và failure cost giảm nhẹ. Đây là hậu quả dịch vụ/trajectory, không phải backlog: không episode nào có overdue cost trong dữ liệu.

Phân phối paired saving so với myopic:

| Method | Thắng / bằng cost / thua, 128 seed | Trajectory hoàn toàn giống myopic | Saving nhỏ nhất / lớn nhất |
|---|---:|---:|---:|
| Joint | 27 / 84 / 17 | 78 seed | −69 / +140 |
| Assignment | 2 / 112 / 14 | 108 seed | −69 / +16 |

Các seed bằng cost không nhất thiết có cùng action/state path; số identical path được kiểm tra riêng trên toàn 60 tick. Joint không cải thiện đồng đều từng seed; mean lợi ích đến từ một nhóm trajectory có upside lớn.

Trên 6.144 tick H mỗi method, joint khác myopic ở **110 tick (1,79%)**: **68 đổi nhóm máy**, **42 chỉ đổi assignment**. Có 41 đổi cardinality và 16 đổi WAIT; hai loại này là tập con/chồng lấp với đổi nhóm máy, không cộng thành 68. Assignment ablation đổi ở **50 tick (0,81%)**, không đổi nhóm máy, cardinality hay WAIT.

Kết quả ablation cho thấy cho phép thay đổi nhóm máy/timing/WAIT có giá trị so với chỉ được đổi technician trong implementation này. **Không lấy chênh lệch hai mean để quy một tỷ lệ chính xác cho từng loại quyết định:** từ lần đổi action đầu, các trajectory và nhóm máy được chọn ở tương lai đã khác. Assignment ablation giữ subset của myopic trên state hiện tại của chính nó, không giữ lịch chọn máy của baseline cả episode.

## Diagnostic cửa sổ 12 tick và return 60 tick khác nhau

Tại 896 state trên trajectory myopic, joint đổi 19 root, assignment đổi 5 root. Đánh giá bằng bank độc lập sau chọn:

| Root policy | Gain 12 tick trung bình / landmark, gồm cả root không đổi | CI 95% khám phá |
|---|---:|---:|
| Joint | +0,1420 | [+0,0350; +0,3126] |
| Assignment | +0,01256 | [+0,00112; +0,03041] |

Đây là cost forecast 12 tick với continuation myopic, không phải return của policy replan 48 tick + common tail. Chỉ năm landmark assignment đổi, nên không diễn giải diagnostic nhỏ dương thành lợi ích assignment toàn episode. Cũng không cộng các gain landmark để dự báo total savings.

Một **ví dụ hậu nghiệm** cho thấy rõ bẫy này: seed **36000083**, state tuổi `[1,3,3,2]`, hai technician rảnh, không máy hỏng. Myopic chọn T0→M1, T1→M2; lookahead đổi thành T1→M1, T0→M2. Forecast 12 tick cho cost 102 và 101. Bank holdout cũng cho gain **+1, SE=0** trong 128 scenario đã dùng, nên ở state này nhiễu scenario không giải thích lựa chọn sai dài hạn.

Cả joint và assignment trở lại **đúng cùng toàn bộ state** ở tick 12,24,36 và lặp lại quyết định đổi assignment. Trace thật ghi:

| Cost tích lũy | Myopic | Joint và assignment |
|---|---:|---:|
| Sau 12 tick | 102 | **101** |
| Sau 24 tick | **186** | 202 |
| Sau 48 tick | **354** | 404 |
| Sau 60 tick, gồm common tail | **438** | 506 |

Myopic ổn định ở các block sau với cost 84/12 tick; lookahead quay lại state cũ và tiếp tục block 101. Lookahead thắng cửa sổ đầu 1 đơn vị nhưng cuối episode thua 68. State `[1,3,3,2]` chiếm 32/50 lần đổi assignment của ablation; đây là pattern thường gặp trong log đổi action, không phải một ví dụ hoàn toàn biệt lập. Tuy vậy chưa có phép can thiệp horizon/value mới để quy toàn bộ mức thua cho một nguyên nhân duy nhất.

**Suy luận từ trace và code:** cắt forecast ở 12 tick, không có value cho state cuối cửa sổ, cùng giả định continuation myopic có thể đánh giá lệch khi policy thật liên tục replan. Tăng số scenario đơn thuần không xử lý được pattern SE=0 này. Không có bằng chứng myopic tối ưu dài hạn, cũng không có bằng chứng mọi lookahead assignment đều kém.

## Hướng tiếp theo

Giữ setting này ở vai trò khám phá và ưu tiên kiểm tra chất lượng planner trước khi quét setting nữa. Một follow-up hợp lý là lookahead có continuation/terminal value phản ánh cost sau cửa sổ, hoặc horizon dài hơn, kèm kiểm tra tránh chu kỳ vừa quan sát. Cần protocol và panel mới; không dùng lại panel này làm xác nhận cho algorithm sửa sau khi xem kết quả. Đây là đề xuất, chưa triển khai/chạy.

Hiện có headroom cho joint decision ở mức vài phần trăm, nhưng chưa vượt practical gate và assignment-only planner hiện tại không có lợi ích. Chưa đủ cơ sở lấy setting này làm benchmark chứng minh cải tiến assignment của RL, và khoảng cách planner không phải ceiling hay mức lợi ích RL sẽ học được.

## Fallacy scan — 11/11 đã kiểm tra

| Loại | Kiểm tra / giới hạn |
|---|---|
| Simpson | Mean đi cùng paired win/tie/loss và từng method; không nói mọi seed cải thiện. |
| Ecological | Tách giá trị policy/trajectory khỏi giá trị assignment tại một state. |
| Berkson | Setting được chọn khám phá từ v3; inference chỉ cho fixed setting và methods đã khóa. |
| Collider | Các tick đổi action/state đặc biệt là diagnostic điều kiện theo policy, không suy hiệu quả toàn quần thể từ chúng. |
| Base rate | Rate toàn H dùng 6.144 tick/method; landmark gain dùng cả 896 state, kể cả root không đổi. |
| Regression to mean | Panel 128 seed mới, không dùng các seed screening v3 để xác nhận. |
| Survivorship | Đủ 384 episode; giữ cả thua và tie, không lọc episode ít/no failures. |
| Look elsewhere | Năm CI chính 99% đã khóa; hai mechanism CI 95% là khám phá; seed ví dụ hậu nghiệm. |
| Forking paths | Giữ margin 1%, confidence, budget và scientific status; không retune bằng full outcomes. |
| Correlation/causation | Paired policy comparison không phân rã nhân quả riêng subset/timing/assignment; forecast gain không là full return. |
| Reverse causality | Policy làm đổi trajectory và tần suất contention; rate không phải thuộc tính cố định của setting. |

## Tái lập

`analyze.py`, `analysis.json` và `validation.log` cùng thư mục chứa toàn bộ kiểm chứng và số liệu chi tiết. Script chỉ đọc run đã tải về, dùng checkout gắn commit, mở rộng audit mọi forecast đã ghi và tính lại thống kê; không huấn luyện, quét setting, mở seed mới hay chạy lab.
