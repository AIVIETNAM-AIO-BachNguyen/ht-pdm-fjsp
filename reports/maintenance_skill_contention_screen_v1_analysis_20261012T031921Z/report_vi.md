# Phân tích maintenance_skill_contention_screen_v1

**Kết luận:** `overlap3__tradeoff_2_4` thắng và đạt cả bốn tiêu chí xác nhận đã đăng ký. Tuy nhiên, thước đo conflict có điểm yếu khi phá hòa: **52–65% conflict được báo ở setting này có thể giải quyết chỉ bằng cách chọn technician khác có cùng điểm**. Setting này là ứng viên cho bài toán đánh đổi tốc độ và mức phục hồi; chưa đủ bằng chứng để chốt là setting gây tranh chấp kỹ năng mạnh nhất.

Material Passport: **ANALYZED**. Phạm vi **VERIFIED** là tính toàn vẹn artifact, tái dựng simulator và kết quả theo hai rule đã cố định. Đây là sàng lọc setting, chưa huấn luyện RL và chưa chứng minh lợi ích của thuật toán RL.

## Dữ liệu và kiểm tra

Run: `maintenance_skill_contention_screen_v1_full_cpu_20261012T031404Z`, commit `8047eeb104dc9b9c3e833ae98b554e13d90a96b3`, checkout sạch lúc chạy. Có 25 setting: 20 ứng viên và 5 đối chứng. Screening dùng 64 seed, confirmation dùng 128 seed mới cho winner, runner-up và hai đối chứng; mỗi setting chạy hai rule.

**PASS:** 15 hash artifact; đủ 3.200 episode screening và 1.024 episode confirmation; seed audit; selection đã khóa; replay 6.960 transition có log; tái dựng toàn bộ **4.224 episode / 253.440 transition**, mọi metric lưu đều khớp. Tái tính độc lập các khoảng tin cậy của bốn gate cũng khớp. Không có phân công thực thi sai skill hoặc dùng trùng technician. Dữ liệu gốc được giữ nguyên.

## Screening đã chọn gì?

Mỗi tỷ lệ dưới đây dùng toàn bộ 48 tick của episode làm mẫu số, lấy kết quả thấp hơn trong hai rule. “Có lựa chọn” nghĩa là có thể phân công cùng tập máy cho technician khác để thay đổi thời gian hoặc mức phục hồi. Nó không chỉ đếm đổi nhãn technician.

| Setting | Conflict theo định nghĩa cũ | Có lựa chọn vật lý | Kết quả screening |
|---|---:|---:|---|
| overlap3__tradeoff_2_4 | 32,23% | 10,97% | Winner |
| balanced2__tradeoff_2_4 | 24,90% | 7,36% | Runner-up |
| overlap3__paired_2_4 | 10,71% | 16,93% | Đạt gate, xếp thứ ba |
| bottleneck2__tradeoff_2_4 | 27,83% | 5,63% | Đạt gate, xếp thứ tư |
| bottleneck1__common_2_4 | **46,35%** | **0,29%** | Không đạt gate lựa chọn |

Có 4/20 ứng viên đạt gate. Ranking đã đăng ký là tích conflict × tỷ lệ có lựa chọn, kèm gate availability và pending; không xếp theo cost. Setting conflict cao nhất thường ép gần như mọi máy về cùng technician, nên ít cơ hội chọn phân công thực sự. Conflict cao nhất không đồng nghĩa benchmark phân công tốt nhất.

## Winner qua xác nhận theo protocol gốc

| Metric | Ước lượng bất lợi hơn giữa hai rule | CI 98,75% | Gate |
|---|---:|---:|---|
| Conflict cũ | 31,05% | 28,78–33,38% | ≥10%: đạt |
| Có lựa chọn vật lý | 10,47% | 8,76–12,27% | ≥5%: đạt |
| Availability | 49,72% | 49,37–50,05% | ≥25%: đạt |
| Máy hỏng chưa được phục vụ tại H | 0,406 | 0,281–0,539 | ≤1: đạt |

Bootstrap 20.000 lần theo seed, giữ ghép cặp hai rule; CI 98,75% cho từng gate theo hiệu chỉnh bốn gate của protocol. Availability bao gồm thời gian máy ngừng vì bảo trì/sửa chữa. Pending khác với failed: máy đang sửa vẫn có thể thuộc failed nhưng không thuộc pending.

Winner có 4 máy, 2 technician, một máy chỉ technician 1 làm được và ba máy có thể chọn cả hai. Cấu hình dùng tên máy/technician bắt đầu từ 1:

| Máy | Technician 1: thời gian / phục hồi | Technician 2: thời gian / phục hồi |
|---|---|---|
| 1 | 2 tick / 50% | Không đủ skill |
| 2 | 4 tick / 100% | 2 tick / 50% |
| 3 | 2 tick / 50% | 4 tick / 100% |
| 4 | 4 tick / 100% | 2 tick / 50% |

“Phục hồi 50%” là mức giảm tuổi máy theo mô hình; không phải xác suất sửa thành công. Lựa chọn nhanh nhưng phục hồi ít và chậm nhưng phục hồi nhiều là khác biệt vật lý thật.

## Điểm yếu của metric conflict

Metric cũ cho mỗi máy tự chọn technician có điểm `restoration / duration` cao nhất trong số đang rảnh. Nếu hai máy chọn cùng người thì ghi conflict. Khi điểm bằng nhau, code chọn technician có chỉ số thấp hơn.

Ở winner, **0,5 / 2 = 1 / 4 = 0,25**. Vì vậy, hai technician khác nhau về tốc độ/phục hồi nhưng hòa điểm heuristic. Nhiều máy cùng chọn technician 1 chỉ vì cách phá hòa. Đây là proposal giả định trước bước matching/reservation; rule thực thi vẫn phân công hợp lệ.

Phân tích bổ sung sau khi xem kết quả giữ tất cả technician đồng điểm tốt nhất cho mỗi máy, rồi kiểm tra có thể ghép chúng với các technician khác nhau hay không:

| Confirmation winner | deadline_matching | priority_greedy |
|---|---:|---:|
| Conflict cũ | 34,70% tick | 31,05% tick |
| Conflict giải được bằng đổi lựa chọn đồng điểm | 22,46% tick | 16,19% tick |
| Tỷ trọng phần trên trong conflict cũ | **64,73%** | **52,15%** |
| Conflict còn lại trong tập lựa chọn tốt nhất | 12,24% tick | 14,86% tick |
| Tập ưu tiên làm giảm số máy có thể phục vụ so với mọi cạnh hợp lệ | 0% | 0% |

Hai phần ở giữa cộng đúng bằng conflict cũ trên mọi episode. Conflict còn lại vẫn bao gồm thiếu technician đang rảnh và giới hạn skill; không thể gọi toàn bộ phần đó là tranh chấp do sở thích kỹ năng.

CI mô tả 98,75% cho tỷ lệ conflict còn lại thấp hơn giữa hai rule là **9,85–14,78%**, nên chưa cho thấy biên dưới vượt 10%. Đây là **diagnostic hậu nghiệm**, không phải gate mới được đăng ký và không thay thế quyết định `CONFIRMED` gốc. Tỷ lệ “tập ưu tiên làm giảm số máy phục vụ” bằng 0 là hệ quả cấu trúc của heuristic hòa điểm ở setting này, không chứng minh RL không có lợi ích. RL vẫn có thể học giá trị dài hạn của nhanh/chậm, thời điểm bảo trì và cách dành technician.

## Chi phí và ý nghĩa với RL

Ở winner, cost trung bình extended horizon của matching là **983,13**, greedy là **1.000,19**: giảm **1,71%**, CI ghép cặp 95% **0,92–2,54%**. Đây là so sánh thăm dò, chưa hiệu chỉnh nhiều so sánh. Hai rule khác cả tập máy được phục vụ lẫn phân công, nên mức giảm không đo riêng lợi ích assignment và không phải mức tăng của RL.

Đối chứng technician bằng nhau vẫn có chênh lệch cost khoảng 0,61%; đối chứng partitioned có hai rule trùng kết quả. Điều này phù hợp với việc thứ tự chọn/reserve máy cũng ảnh hưởng cost. Trong winner, technician utilization khoảng 94–96%, nên áp lực công suất đáng kể; không nên gán mọi khó khăn cho skill.

## Quyết định đề xuất

Giữ `overlap3__tradeoff_2_4` làm **ứng viên đánh đổi tốc độ–phục hồi**. Chưa chốt nó là benchmark “conflict kỹ năng mạnh nhất”. Nếu mục tiêu vẫn là tranh chấp kỹ năng, bước kế tiếp nên định nghĩa metric không phụ thuộc phá hòa, phân biệt thiếu công suất với cạnh tranh technician ưu tiên, rồi đăng ký screening/confirmation trên seed mới trước khi xem kết quả.

Các setting cùng thích technician nhanh như `overlap3__common_2_4` có tín hiệu cạnh tranh ưu tiên rõ hơn trong diagnostic screening, nhưng không đạt ngưỡng 5% cơ hội thay đổi phân công đã đăng ký. Không đổi winner sang chúng sau khi xem test. Hai rule trên lưới hữu hạn này cũng chưa xác định cực đại toàn cục của “conflict” hay bảo đảm độ khó cho một policy RL khác.

Không huấn luyện RL hoặc chạy experiment mới trong lượt phân tích này.

## Kiểm tra 11 sai lệch suy luận

1. Simpson: giữ số của hai rule riêng; phép lấy min/max được nêu rõ.
2. Ecological: không coi tỷ lệ trung bình là đặc điểm mọi trạng thái hay mọi policy.
3. Berkson: winner là kết quả chọn trên lưới thiết kế; không đại diện mọi setting.
4. Collider: giữ mọi episode, không lọc theo thành công hay cost.
5. Base-rate: mẫu số là 48 tick; phân biệt tỷ lệ tick và tỷ trọng trong conflict.
6. Regression to mean: xác nhận 128 seed mới; diagnostic mới không thành test đăng ký trước.
7. Survivorship: giữ pending, failed, tail và mọi trường hợp không phục vụ được.
8. Look-elsewhere: 25 setting là screening; bốn gate xác nhận có hiệu chỉnh; cost/diagnostic được ghi hậu nghiệm.
9. Forking paths: không đổi gate, rerank hoặc chọn winner thay thế sau confirmation.
10. Correlation/causation: rule gap không chứng minh lợi ích riêng assignment hoặc RL.
11. Reverse causality: cấu hình/rule cố định trước outcomes; không suy tác dụng của skill chỉ từ tương quan metric.

## Tái lập phân tích

`analyze.py` kiểm tra hash, binding mã nguồn với commit, replay, ranking, CI và tạo JSON/CSV diagnostic. `diagnostic_intervals.py` tái lập CI hậu nghiệm từ CSV đã tái dựng. Runtime dùng NumPy/tqdm trong checkout mã nguồn tại `/private/tmp/marl-maintenance-assignment-confidence-v1`; file phân tích ghi rõ đường dẫn và commit ràng buộc. Kết quả máy đọc: `analysis.json`, `tie_diagnostics_episodes.csv`, `tie_diagnostics_means.csv`, `tie_diagnostic_intervals.json`; bằng chứng kiểm tra: `validation.log`.
