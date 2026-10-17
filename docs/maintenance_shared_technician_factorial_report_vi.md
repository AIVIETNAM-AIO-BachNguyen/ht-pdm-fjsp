# Báo cáo: yêu cầu bảo trì và phân công technician dùng chung

Ngày 2026-10-18 (Asia/Taipei). Hai nghiên cứu hoàn tất: v1 có protocol khóa
trước khi mở test; v2 là follow-up **exploratory** được viết sau khi thấy một
giới hạn cấu trúc của v1. Không gộp hai panel hay diễn giải smoke thành kết quả
khoa học.

## Kết luận ngắn

Trong năm cấu hình tổng hợp đã thử, tạo yêu cầu phòng ngừa trước khi máy hỏng
giảm tổng chi phí rất mạnh so với chỉ yêu cầu sau failure. Nhưng việc ghép
quy tắc yêu cầu `capacity_aware` với dispatcher `lookahead` **không có bằng
chứng giảm chi phí thêm** so với các quyết định đơn giản hơn. FCFS xét kỹ năng
thường là đối chứng mạnh. V2 cho thấy khi service kéo dài nhiều tick, yêu cầu
sớm hơn kết hợp với FCFS có thể giảm chi phí nhỏ: 3,57% trong cấu hình balanced;
đây là kết quả exploratory trên một ma trận cố định, chưa xác nhận tính tổng
quát. Không có bằng chứng cho tương tác timing × allocation có lợi trong các
cấu hình đã chạy.

## Framework và phép so sánh

Tái sử dụng `maintenance_dispatch`/`maintenance_waiting`: N4 máy, kỹ năng
machine × technician, technician không thể phục vụ hai máy cùng lúc, service
1–5 tick tùy cạnh, PM/CM, failure tape cố định theo seed, restoration chỉ khi
hoàn tất. Không có production scheduling. Code mới thêm pool yêu cầu bền vững:
proposal tạo pending demand; dispatcher chỉ nhận demand pending; khi nhận,
job giữ technician đến completion; máy khỏe đang chờ tiếp tục hoạt động và
degrade, máy hỏng đang chờ unavailable. PM pending biến thành CM khi máy hỏng
nhưng giữ thời điểm yêu cầu. Failure tạo demand tự động nếu chưa có; không
reset episode hay xóa nhu cầu khi technician bận. Ở H48, mọi arm dùng cùng
FCFS trong 12 tick cleanup; nhu cầu/công việc còn lại chịu terminal liability
đã khóa trong protocol. Chi phí chính = PM/CM + unavailable + failure + chờ
quá deadline trong H+tail + terminal liability. Cùng dynamics và cost cho mọi
arm; policy chỉ thấy state hiện tại và xác suất failure, không thấy shock tương
lai.

Factorial 3 request rules (`reactive`, `threshold`, `capacity_aware`) × 4
dispatchers (FCFS xét kỹ năng, risk priority, myopic matching, six-tick
isolated-machine lookahead matching). Lookahead tối ưu partial matching theo
bid tính từ DP từng máy, **không** là oracle tối ưu chung: nó giả định sau tick
hiện tại technician luôn sẵn và có thể đánh giá sai chi phí chiếm dụng dài
ngày. Myopic chỉ dùng lợi ích/cost trước mắt. Không huấn luyện RL; train seeds,
checkpoint và training replications không áp dụng. Đơn vị thống kê là 64
physical test seeds ghép cặp trong mỗi cấu hình; 12 arm chia sẻ initial state,
shock tape và cost. Bootstrap resample cả vector 12 arm của mỗi seed 10.000
lần. CI v1 98,75% theo protocol, CI v2 95% descriptive.

V1 gồm `scarce_overlap` K2, `partitioned` K2, `ample` K4. V2 cố định thêm
`overlap_service2` và `balanced_service2`, đều K2 với service hợp lệ tối thiểu
2 tick. Ma trận và budgets cụ thể ở protocol, snapshot trong artifacts. Không
đổi seed, threshold, metric hay stopping rule sau khi đọc mỗi panel. Không có
oracle joint chính xác: DP cũ của repository không bao gồm pool yêu cầu bền
vững; áp dụng trực tiếp sẽ đổi bài toán. Vì vậy không có claim về khoảng cách
tới optimum.

## Gate, provenance và tài nguyên

| Gate | V1 confirmatory | V2 exploratory |
|---|---:|---:|
| Protocol | `maintenance_shared_technician_factorial_v1_protocol.md` | `maintenance_shared_technician_exploratory_v2_protocol.md` |
| Code push trước full | `a7fb734` | `a6f702b` |
| Tests/smoke | 998 regression pass; smoke 72/72, replay 4.320 tick | 21 targeted pass; smoke 48/48, replay 2.880 tick |
| Full | 2.880/2.880 episode, 33,87 s CPU | 1.920/1.920 episode, 21,80 s CPU |
| Test trace | 2.304 episode, 138.240 tick replay PASS | 1.536 episode, 92.160 tick replay PASS |
| Manifest/integrity | `COMPLETED` / `PASS` | `COMPLETED` / `PASS` |

V1 train = rỗng, development 91000100–115, sealed test 91000200–263;
v2 train = rỗng, development 93000100–115, test 93000200–263. Không overlap
giữa các split hoặc các registry đọc được. Mỗi full run ghi partial/final CSV,
trace, summary, config, source/protocol/registry hash và Git commit trong thư
mục timestamp mới. `invalid_assignments=0`, matching/skill/capacity hợp lệ,
request conservation, H+tail+liability và component cost đều đối chiếu; trace
được replay từ shock tape theo seed. Đây là kiểm tra tính hợp lệ, **không** là
bằng chứng chính sách tối ưu. Một smoke v1 lỗi ghi thứ tự trường trace được
giữ nguyên và sửa trước commit/full; dynamics và cost không bị thay đổi.

## Kết quả v1: panel đã khóa

Cost trung bình/episode, 64 test seeds. Δ = `capacity_aware/lookahead` trừ
`threshold/FCFS`; số âm có lợi cho arm phối hợp. CI 98,75% cho Δ.

| Điều kiện | reactive/FCFS | threshold/FCFS | capacity-aware/FCFS | capacity-aware/lookahead | Δ [CI 98,75%] |
|---|---:|---:|---:|---:|---:|
| Scarce overlap | 1.096,14 | 644,20 | 644,20 | 650,89 | +6,69 [−6,92; +19,63] |
| Partitioned | 1.174,61 | 854,53 | 851,86 | 854,50 | −0,03 [−15,96; +16,04] |
| Ample | 1.074,34 | 549,98 | 549,98 | 741,78 | +191,80 [+164,11; +217,60] |

**H1 (joint giảm ≥5% so với cả hai control riêng): NOT SUPPORTED.** Ở scarce,
allocation contrast +6,69 (CI 98,75% [−6,92; +19,63], effect size ghép cặp
0,157, chỉ 10/64 seeds thắng); request contrast đúng 0. **H2 (tương tác âm
≥2%): NOT SUPPORTED**; interaction bằng 0, CI [0;0] ở scarce. **H3:
INCONCLUSIVE về khái quát hóa**, vì protocol chỉ đăng ký so sánh mô tả trên
ba ma trận cố định; không thấy joint gain ổn định và ở ample lookahead tăng cost
34,87% so với threshold/FCFS. `threshold/FCFS` giảm cost so với reactive/FCFS
lần lượt 41,23%, 27,25%, 48,81%; 95% paired CI mô tả cho chênh lệch cost có
trong supplementary analysis. Đây là lợi ích của preventive request timing
so với đợi failure, không chứng minh lợi ích của coordination thêm.

Trace giải thích vì sao H2 v1 không có cơ hội biểu hiện ở điều kiện chính:
`scarce_overlap` có technician chung nhanh duration 1, nên cả hai technician
không cùng bận ở ranh giới quyết định; `capacity_aware` không tạo proposal
khác `threshold` ở 0/3.072 policy ticks trên mỗi dispatcher. `partitioned`
có nhiều proposal sớm hơn nhưng 0 assignment khác trong H; lợi ích FCFS chỉ
đến từ cleanup tail (−2,67 cost/episode), không từ hành động trong H. Ở ample,
lookahead làm unavailable tăng rõ dù failure gần bằng: availability H 57,28%
so với 75,00% của threshold/FCFS, vì nhiều PM/service hơn. Availability tính
cả thời gian PM nên không đi cùng chiều cost trong mọi contrast.

## V2 exploratory: service kéo dài nhiều tick

CI 95% dưới đây là mô tả trên 64 seed mới, không là kiểm định xác nhận v1.

| Điều kiện | threshold/FCFS | capacity-aware/FCFS | Δ timing [CI 95%] | capacity-aware/lookahead | Δ allocation so với capacity-aware/FCFS [CI 95%] |
|---|---:|---:|---:|---:|---:|
| Overlap service2 | 981,45 | 974,05 | −7,41 [−16,73; +1,64] | 995,72 | +21,67 [+5,12; +38,39] |
| Balanced service2 | 870,36 | 839,25 | −31,11 [−47,92; −16,37] | 1.095,13 | +255,88 [+223,31; +287,91] |

Ở balanced, timing+FCFS giảm 3,57% so với threshold+FCFS; effect size paired
cho Δ timing = −0,480. Nó giảm H cost 29,19, tail 3,89, nhưng tăng terminal
liability 1,97. Availability H tăng 1,41 điểm phần trăm, failures giảm 0,98
và failed waiting giảm 1,97 tick/episode. Ở overlap, timing giảm 0,75%, CI
cắt 0. Hai cấu hình này có technician bận qua nhiều tick: dưới FCFS,
`capacity_aware` đổi proposal/assignment ở 720/354 ticks (balanced) và
1.641/1.079 ticks (overlap) trên 3.072 tick test mỗi điều kiện. Với
lookahead, proposal đổi nhưng assignment trong H không đổi ở cả hai cấu hình;
dispatcher này thường trì hoãn demand và không chuyển tín hiệu timing thành
hành động. Balanced lookahead làm failed waiting tăng 28,92 tick/episode và
overdue tăng 6,09 tick/episode so với capacity-aware/FCFS. Interaction
balanced ngược hướng mong đợi: +28,92 cost, CI 95% [+14,39; +45,56].

Theo ba câu hỏi exploratory của v2: **E1 không quan sát được** (joint lookahead
đắt hơn capacity-aware/FCFS ở cả hai điều kiện); **E2 không quan sát được**
(interaction balanced dương, overlap CI cắt 0); **E3 quan sát được** với FCFS
qua các assignment khác trong H. Đây không phải ba kết luận confirmatory.

## Diễn giải và giới hạn

Trong các điều kiện đã kiểm tra, phối hợp **tạo request sớm khi tài nguyên
bận + FCFS** đôi lúc có lợi nhỏ. Không thấy lợi ích của dispatcher lookahead
trên cùng cơ chế; risk và myopic là các đối chứng đã chạy đầy đủ và mọi cost
theo arm nằm trong `summary.json`. Vì lookahead là DP từng máy, kết quả xấu
không loại trừ một lookahead chung tính đúng opportunity cost hoặc một chính
sách học. Không có bằng chứng rằng MARL hay giao tiếp là cần thiết. Kết quả
âm của v1 là kết luận đúng với panel khóa; v2 không biến nó thành positive
confirmatory result.

Giới hạn chính: chỉ năm ma trận tổng hợp, một failure law và cost schedule,
H48+cleanup12, terminal liability xấp xỉ, action tại ranh giới tick, không
có production loss riêng, không đo độ trễ truyền thông hoặc tổng quát hóa qua
plant. V1 scarce có skill competition nhưng thiếu busy kéo dài cho timing
rule; v2 sửa đúng giới hạn cấu trúc này song được khởi xướng sau khi mở v1
test. CI phản ánh biến thiên shock/initial state theo seed cố định, không
phản ánh biến thiên cấu hình plant hay training seed. Runtime CPU là toàn bộ
runner/audit, không phải latency từng dispatcher. Chưa có phép so sánh với
joint oracle tối ưu.

## Artifacts và tái lập

- V1 full: `artifacts/maintenance_shared_technician_factorial_v1_full_cpu_20261017T172615Z/`
- V2 full: `artifacts/maintenance_shared_technician_exploratory_v2_full_cpu_20261017T172923Z/`
- Supplementary analysis (hash input, exposure, paired metrics):
  `artifacts/maintenance_shared_technician_analysis_20261017T173145Z/supplementary_analysis.json`
- Code phân tích: `scripts/analyze_shared_technician_factorial.py`.

Lệnh Mac để lấy đúng ba artifact trên, gồm cả full logs, được nêu trong câu
trả lời cuối; chỉ người dùng chạy `rsync -avhP --partial`. Không có transfer
hay SSH nào được thực hiện bởi Codex.
