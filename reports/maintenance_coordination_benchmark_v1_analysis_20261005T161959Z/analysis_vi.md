# Phân tích maintenance coordination benchmark v1

Ngày phân tích: 06/10/2026, Asia/Taipei. Run:
`maintenance_coordination_benchmark_v1_full_cpu_20261005T142122Z`.
Commit `0cafebbfb54bd9835dac210d9435befcc9c42f42`, checkout sạch,
protocol full đã khóa. Không thay raw data, protocol, seed, checkpoint hoặc baseline.

**Kết luận chính: panel này không cho thấy lợi ích tổng cost của critic chung
hay message. MARL có message đắt hơn deadline matching 3,61%, với chênh lệch
cost dương trên cả 10 training seed.** Đây là bằng chứng bất lợi cho thiết kế
MARL đang thử ở ngân sách này, mạnh hơn việc chỉ nói “chưa đủ bằng chứng”.
Cả bốn giả thuyết confirmatory đều FAIL. Không suy rộng thành mọi MARL đều vô ích
hoặc maintenance không có bài toán tối ưu.

## 1. Dữ liệu và độ tin cậy

Run COMPLETED trong khoảng 75 phút tính từ manifest started_at đến completed_at.
Đủ 50 model, 6.000.000 transition, 500.000 episode training, 200.000 optimizer
step. Đủ 20.000 learned test +1.200 reference test +4.000 message-off test;
1.000 learned development +60 reference development; 53 exact rows.
Có 97.680 machine-episode records bao gồm development.

21 engineering audit của runner đều đạt. Tôi chạy lại audit trên Mac:
**25.200 test episode /318.150 tick** khớp checkpoint, sequence, proposal,
physical matching và chi phí. Có 252.500 tick thuộc các proposal arm.
Không có mismatch policy giữa checkpoint CPU Ubuntu và replay CPU Mac trên
panel này. Tự tính arbiter từ reported urgency và tự tính cost scalar đều khớp;
per-machine starts/wait/unavailability khớp tổng episode. Source bundle SHA256
`14956b3faac576ad81d5594504c527b208f7d32d68a338fea5abd8a582147bb6` đúng.

Script phân tích độc lập kiểm tra cả 500.000 training record: thứ tự episode,
12 bước/episode, từng model đủ120.000 bước, cost thành phần và overdue price12
đúng; physical population hash khớp cả năm arm cho từng training seed.
Kiểm tra raw panel keys, checkpoint counts, progress logs hữu hạn, source hashes,
CI/effect/wins/gates tính lại từ raw CSV đều khớp summary. Có hash inventory
của mọi file raw trong report. Đây là kiểm tra logs/cost/budget training;
không phải replay toàn bộ6m training transition hoặc từng optimizer update.

## 2. Kết quả chính

Primary score là trung bình đều bốn family N3/N4 nominal/pressure, mỗi family
50 evaluation seed. Đơn vị suy luận là 10 training seed. Deadline matching được
chọn bằng **development**, trước reference test; development cost105,80 so với
risk rule108,05 và lookahead114,60; cả ba đạt p95 wait<=4 trên development.
Không đổi sang baseline khác sau khi nhìn test.

| Phương pháp | Base cost | Base + overdue price | Nominal base cost |
|---|---:|---:|---:|
| Deadline matching | 148,8800 | 148,9400 | 93,9400 |
| PPO tập trung autoregressive | 149,4675 | 149,5635 | 95,6690 |
| IPPO local | 153,8855 | 153,9035 | 98,2480 |
| MAPPO local | 153,9505 | 153,9505 | 98,5020 |
| MAPPO message | 154,2535 | 154,2895 | 99,9080 |
| Full-information proposal | 154,7400 | 154,7520 | 100,2680 |
| Risk/skill rule | 150,9750 | 150,9750 | 94,5200 |
| Isolated lookahead matching | 149,4250 | 153,2050 | 95,4400 |

Bảng nominal các reference chỉ để đọc tradeoff; raw/family CSV là nguồn chuẩn.

| Giả thuyết | Chênh lệch cost tương đối | Thắng seed | CI98,75% của chênh lệch cost | Kết quả |
|---|---:|---:|---:|---|
| H_learning: central_ar−deadline, base | +0,395% | 3/10 | [−0,555; +1,730] | FAIL |
| H_multiagent: message−deadline, base | +3,609% | 0/10 | [+2,367; +8,380] | FAIL |
| H_critic: MAPPO local−IPPO, priced | +0,031% | 5/10 | [−1,569; +1,663] | FAIL |
| H_message: message−MAPPO local, priced | +0,220% | 6/10 | [−2,441; +3,119] | FAIL |

Dương nghĩa là tốn hơn. “Priced”=base cost+12×overdue ticks, cùng định nghĩa
cho mọi phương pháp. CI là paired Student-t(df9), Bonferroni bốn contrast.
Các arm vẫn đạt p95 wait<=4 ở từng pressure family và từng training seed,
nominal/base cost không xấu quá10%. FAIL nằm ở hiệu quả>=5%, >=8/10 wins và
uncertainty, không phải do dữ liệu thiếu hay guard nominal tự động loại model.

Nếu chia CI chênh lệch cho mean control để dễ đọc: central PPO nằm khoảng
[−0,37%; +1,16%], critic [−1,02%; +1,08%], message [−1,59%; +2,03%].
Mức giảm5% nằm ngoài cả ba khoảng này trên panel đã thử. Không nên giải thích
mọi FAIL là chỉ cần thêm vài seed. Các khoảng phần trăm này là CI **chênh lệch
cost được scale bằng mean control**, không phải một CI ratio độc lập.
Uncertainty phản ánh training seed với finite evaluation panel cố định;
không phải bảo đảm cho mọi môi trường ngoài panel.

## 3. Điều có ích nhưng chưa chứng minh nhu cầu MARL

**Lợi ích cục bộ của scheduling dài hạn:** ở N4 pressure, central_ar cost242,582
so với deadline246,700: giảm1,67%,10/10 training seed tốt hơn. CI95% exploratory
của delta [−5,593; −2,643]; đây là đọc riêng một family, không thay kết quả
confirmatory toàn panel. Central_ar tạo nhiều PM hơn (4,942 vs3,800), ít failure
hơn (3,912 vs5,060), nhưng unavailable ticks cao hơn (28,832 vs26,400).
Delta cost=−1,490 maintenance−17,220 failure+14,592 downtime=−4,118.
Policy học một tradeoff thật; mức lợi ích này đến từ central scheduler và
chưa đạt tiêu chí practical5% đã khóa.

**Message cải thiện một diagnostic nhưng không thành giảm cost:** các yêu cầu
đã gửi, có thể ghép thêm với worker idle bằng matching khác, nhưng bị bỏ lại:
MAPPO local0,678→message0,4475/episode, giảm34,0%. Tuy nhiên rejected proposals
2,782→3,2555/episode tăng17,0%; tổng priced cost153,9505→154,2895.
Số request có thể ghép thêm là maximum-cardinality diagnostic. Phục vụ thêm
chưa chắc tối ưu cost; chưa có counterfactual chứng minh những tick đó gây loss.
Không thể dùng riêng tỷ lệ34% để khẳng định impact kinh tế của communication.

**Thông tin toàn cục chưa giúp proposal policy:** full-information proposal
priced cost154,752 vslocal MAPPO153,9505; delta+0,8015, CI95% exploratory
[−1,709; +3,312]. Nhánh full-information vẫn tốn base cost hơn central_ar3,53%,
0/10 seed tốt hơn, CI95% delta[+2,727; +7,818]. Điều này phù hợp với hạn chế của
simultaneous proposal +fixed arbiter/no retry, nhưng không cô lập nguyên nhân:
encoder, active capacity, factorization, clipping và entropy scaling giữa
central_ar/proposal còn khác. central_ar109.508 parameters; proposal arms76.871.
Cùng parameter count trong bốn proposal arm cũng không đồng nghĩa mọi branch
đều có active capacity giống nhau. Không được quy toàn bộ gap cho collision.

**Tắt message không làm primary score sụp:** cùng checkpoint message-off priced
cost154,1705 vsmessage-on154,2895; delta−0,1190, CI95%[−1,602; +1,364],4/10 seed
gain. Đây là OOD intervention trên checkpoint, không phải control được train
riêng. Nó không chứng minh message không mang thông tin, nhưng không hỗ trợ
claim policy đang phụ thuộc vào message để giảm tổng cost.
Message có6/10 seed thắng nhưng seed128004 có delta priced cost+7,165;
không loại seed này hoặc chọn checkpoint khác sau khi nhìn test.
Message-on có khoảng175,968 scalar packet payload/primary episode; chưa đo
actual network latency, privacy, energy hoặc communication cost toàn hệ thống.

## 4. Waiting, exact và OOD

Ở N4 pressure, central_ar p95wait4, maximum6,0,60% machine records vi phạm B4;
deadline p95wait2,maximum5,0,50% vi phạm; MAPPO local p95wait2,maximum3,0 vi phạm;
message p95wait2,maximum5,0,25% vi phạm. Risk rule cũng có p95wait2,maximum3,
0 vi phạm và basecost250,180, tương đương message250,710. Đạt p95 guard không
có nghĩa là đảm bảo hard bound cho mọi máy. Mọi method ở primary có0 terminal
failed-never-started records; “never started” là cả episode, không phải từ lần
failure cuối nên không diễn giải thành mọi lần hỏng đã được sửa trước terminal.

Ở secondary N5 pressure, central_ar base314,386, p95wait4; deadline322,840,
p95wait5 (trượt B4); risk329,100,p95wait4. Vì có khác biệt feasibility của
reference trên OOD, không coi delta này là thắng confirmatory. Central_ar vẫn
có4,44% machine records wait>4. N4 H24 central_ar495,228 vsdeadline506,920,
giảm2,31%, nhưng vẫn là secondary; proposal/message costs cao hơn cả hai.

Exact N2/K2: economic floor30,500,1475 states,9 initial ages, Bellman residual0.
Central_ar30,6667 (gap0,1667); local MAPPO30,8600; message31,1133;
IPPO31,5635; full-information proposal31,5139; deadline32,4444;
lookahead30,6111. PPO có thể học gần optimum ở bài nhỏ, nhưng một đối chứng
model-based cũng đạt gần optimum. Đây là base-cost diagnostic, không phải
SLA-constrained optimum và không cung cấp ceiling cho N3/N4.

Training costs giảm ở cả năm arm. Ví dụ central_ar first500 mean195,159→last500
138,738; message168,805→141,862. Population đầu/cuối khác nhau và training mix
khác primary test. Các đường này cho thấy có tiến bộ học, không chứng minh
convergence tuyệt đối hoặc full-budget optimum. Các claim trên giới hạn ở
120.000 transition/model, final checkpoint và kiến trúc PPO adaptations này.

## 5. Quyết định nghiên cứu

Bằng chứng cho thấy yêu cầu chung/resource contention là một bài toán scheduling
có tradeoff, nhưng heuristic matching mạnh đã xử lý rất tốt panel hiện tại.
Không có bằng chứng impact kinh tế>=5% từ critic chung/message; MARL message
thua đối chứng chính với CI dương. IPPO tự nó cũng là MARL; “central critic có
ích không” và “MARL cần thiết không” là hai câu hỏi khác nhau.

Tôi sẽ giữ deadline/risk matching làm mốc và ưu tiên xác định hạn chế thực tế
của những rule đó: thông tin không quan sát được, thông tin trễ, khả năng phục
vụ biến động, hoặc quyết định tương tác mà rule mất cost rõ ràng. Các yếu tố này
là hướng cần xác minh từ bài toán thực, không phải tự thêm để làm MARL thắng.
Khi chưa có ràng buộc phân tán thật, central scheduler và matching bids vẫn là
lời giải hợp lệ. Run này không hỗ trợ chuyển sang thêm module MARL hoặc chỉ
kéo dài training để bảo vệ claim necessity. Chưa có cơ sở nói toàn bộ bài toán
không cần học; có cơ sở thu hẹp claim và xác định problem trước khi phát triển
thuật toán mới.

Nguồn trong report: `statistics.json`, `contrasts.csv`, `primary_metrics.csv`,
`family_metrics.csv`, `family_seed_metrics.csv`, `exact_metrics.csv`,
`training_audit.csv`, `replay_audit.json`, `raw_inventory.csv` và hai script
`analyze.py`/`plot.py`. Các contrast D_* là exploratory, CI95% chưa điều chỉnh
multiplicity. Confirmatory contrast giữ nguyên CI98,75% và tiêu chí đã khóa.
