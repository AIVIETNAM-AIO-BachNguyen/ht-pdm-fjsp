# Phân tích maintenance_solver_comparison_v1 — dữ liệu full mới

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-08
- Verification Status: ANALYZED
- Version Label: solver_comparison_validation_v1
- Nguồn: `lab_results/maintenance_solver_comparison_v1_full_cpu_20261007T151955Z`
- Commit thí nghiệm: `4b1cffb0cc1e2413e27f0e2c3a7730f8d3e3d7a8`
- Phạm vi: phân tích lại dữ liệu thô và replay dynamics; không chạy lại training hay actor inference.
- Mức tin cậy: kết luận chắc trong panel đã khóa; CAUTION khi khái quát sang triển khai thực tế hoặc quy mô khác.

## Kết luận cần dùng để quyết định hướng nghiên cứu

**Dữ liệu mới ủng hộ việc chốt bài toán trước khi phát triển thêm kiến trúc MARL.** Có giá trị của quyết định xét tương tác giữa máy; học tập trung đạt chất lượng gần joint planner với thời gian quyết định ngắn hơn. Tuy nhiên, MARL phối hợp hiện tại chỉ cải thiện rõ so với policy độc lập, chưa đem lại lợi ích thêm so với các lời giải tập trung mạnh.

Không nên kết luận “MARL không bao giờ cần thiết”. Kết luận được hỗ trợ là: **với N=4, technician thụ động, cùng simulator và contract matching hiện tại, chưa có bằng chứng để chọn MARL làm hướng mặc định**. Planner và RL tập trung là các đối chứng cạnh tranh thực sự.

## 1. Run và tính toàn vẹn

Run có trạng thái COMPLETED, checkout sạch, chạy CPU một thread. Bản torch mang hậu tố CUDA không có nghĩa thí nghiệm đã dùng GPU. Thời gian toàn run từ 07/10 15:19:56 UTC đến 08/10 10:17:52 UTC, khoảng 18 giờ 58 phút.

| Hạng mục | Dữ liệu thực tế | Kiểm tra lại |
|---|---:|---|
| Artifacts có checksum trong manifest | 352 | Khớp SHA-256 và dung lượng toàn bộ |
| Source snapshot | 27 file | Khớp commit Git và code dùng để replay |
| Model học | 50 = 5 arm × 10 seed | Đủ coverage và lựa chọn checkpoint |
| Training episodes | 2.048.000 | Đọc toàn bộ log, đối chiếu cấu hình giữa arm |
| Training physical steps | 24.576.000 | 491.520/model, khớp progress log |
| Optimizer steps | 512.000 | 10.240/model; 320 update/model |
| Development episodes | 82.560 | Đủ dữ liệu |
| Test episodes | 286.720 | Không trùng khóa episode; đủ grid mỗi controller |
| Test machine rows | 1.146.880 | Bốn máy/episode; cộng lại khớp các chi phí và trạng thái cuối |
| Test decision intervals | 5.160.960 | Replay dynamics, tính hợp lệ, state continuity và chi phí |

Năm arm có cùng chuỗi configuration seed, environment seed, config hash và condition cho từng training seed/episode: 409.600 chỉ mục chung. Mỗi model có 32.768 episode specialized và 8.192 nominal. Không loại seed học khỏi phân tích. Cả 50 lựa chọn checkpoint đều FEASIBLE; checkpoint được chọn là ứng viên có development cost thấp nhất trong các ứng viên thỏa guard. Không chọn checkpoint bằng test.

Sai số đối soát reward được ghi trong log tối đa 2,67×10⁻⁶; tổng machine cost khớp objective. Các boolean audit do chương trình sinh ra không được coi là bằng chứng độc lập: các kiểm tra dữ liệu thô ở trên được thực hiện riêng.

## 2. Bài toán và phạm vi thống kê

Điều kiện chính là **specialized, N=4, K=2, H=12**. Technician khác nhau về thời gian service và restoration; ba máy cùng thích một technician, máy còn lại thích technician kia. Cost gồm maintenance, failure, downtime và giá mềm cho thời gian chờ corrective quá bốn tick.

Đây là simulator maintenance được kiểm soát, chưa phải mô hình sản xuất FJSP đầy đủ hay bộ tham số hiệu chỉnh từ nhà máy. Waiting limit là ngưỡng để tính giá mềm, không phải bảo đảm SLA cứng.

128 profile có nhãn được chia 80 train / 16 development / 32 test trước khi chạy. Test dùng 20 shock seed chung/profile, bốn condition và hai horizon. Profile specialized có nhãn không trùng giữa các split; phép chiếu homogeneous và các profile tương đương sau hoán vị có thể trùng. Vì vậy không được gọi đây là generalization sang mọi cấu trúc vật lý chưa từng thấy.

Đã tính lại cả **96 contrast** từ `episodes.csv`, khớp mean difference, wins và CI trong summary. Bốn contrast chính sử dụng CI hai phía **98,75%**, tương ứng Bonferroni cho family bốn kiểm định. Các sensitivity và contrast phụ dùng CI 95% chưa hiệu chỉnh, chỉ là exploratory.

Đơn vị lặp của contrast có học là **10 training seed**, sau khi lấy trung bình cùng panel test; không coi hàng nghìn episode là các replica training độc lập. CI này mô tả biến động training conditional trên panel profile/shock cố định. Contrast giữa hai planner dùng 32 profile, conditional trên shock panel chung. Không diễn giải CI như đã bao phủ mọi nguồn bất định môi trường.

## 3. Chất lượng trên điều kiện chính

Cost càng thấp càng tốt. Waiting violation là tỷ lệ episode có thời gian chờ vượt ngưỡng, không phải tỷ lệ request.

| Controller | Priced cost | Base cost | Episode có quá hạn | Failures/episode |
|---|---:|---:|---:|---:|
| Rule mạnh: adaptive isolated + matching | 167,695 | 167,227 | 3,750% | 1,950 |
| Joint rollout planner | **162,073** | **162,073** | **0,000%** | **1,720** |
| Fixed-allocation rollout planner | 163,064 | 163,064 | 0,000% | 1,817 |
| Central matching PPO | 163,279 | 163,234 | 0,359% | 2,258 |
| Central fixed-allocation PPO | 163,170 | 163,113 | 0,453% | 2,194 |
| Independent local PPO | 200,335 | 199,909 | 3,141% | 2,650 |
| Cooperative local PPO | 171,795 | 171,233 | 3,781% | 2,709 |
| Cooperative full-information PPO | 171,995 | 171,603 | 2,859% | 2,749 |

So với joint planner, central matching tốn thêm 1,206 cost, tức 0,744%; central fixed tốn thêm 1,096, tức 0,676%. CI 95% của hai chênh lệch lần lượt [0,655; 1,757] và [0,288; 1,904], nhưng đây là so sánh phụ. Joint planner dẫn đầu trong các controller được thử ở điều kiện chính; chưa chứng minh tối ưu toàn cục.

### Bốn câu hỏi đã khóa

Delta = cost bên trái trừ bên phải; delta âm có lợi cho bên trái.

| Contrast chính | Delta | CI 98,75% | Wins | Kết luận theo gate |
|---|---:|---|---:|---|
| Joint planner − isolated valuation + matching | −5,622 | [−7,614; −3,630] | 29/32 profile | **PASS**: priced giảm 3,35%, base giảm 3,08% |
| Central fixed PPO − rule mạnh | −4,526 | [−5,637; −3,415] | 10/10 seed | **FAIL gate độ lớn**: giảm 2,70% priced, 2,46% base; yêu cầu 5% và 3% |
| Cooperative local − independent local | −28,539 | [−29,718; −27,360] | 10/10 seed | **PASS**: priced giảm 14,25%, base giảm 14,34% |
| Cooperative full − central matching PPO | +8,716 | [+7,936; +9,496] | 0/10 seed | **FAIL**: MARL full tốn hơn central 5,34% |

Tất cả left controller trong bốn contrast chính đều qua nominal guard và waiting guard đã khóa. Learning FAIL không có nghĩa RL không học được: mức giảm có CI dưới zero và thắng cả 10 seed, nhưng chưa đủ lớn theo tiêu chí thực dụng đã đặt trước run.

Kiểm tra bổ sung bằng sign-flip hai phía, liệt kê đầy đủ 1.024 mẫu dấu cho các contrast 10 seed, cho p=0,001953 ở ba contrast chính có học. Chiều của effect vẫn phải đọc từ delta: contrast decomposition có bằng chứng theo chiều MARL **kém hơn**. Kiểm tra này cần giả định trao đổi dấu/symmetry; không thay thế protocol hoặc tạo thêm claim confirmatory. Paired-t với n=10 vẫn có giới hạn về giả định phân phối.

Tính mô tả theo 32 profile, sau khi gộp training seed và shock: cooperative local thắng independent 31/32 profile; cooperative full thắng central matching chỉ 1/32; central fixed thắng rule 30/32. Central matching thắng planner 12/32. Đây là consistency check bổ sung, không phải 32 replica training độc lập hay family kiểm định mới.

### Học có diễn ra, nhưng tăng training chưa chắc giải quyết khoảng cách

Mean development cost specialized H12 của central matching giảm từ 395,253 ở initialization xuống 166,963 tại 122.880 step và 166,105 tại 491.520 step. Central fixed giảm 385,109 → 166,919 → 165,693; cooperative local giảm 378,638 → 171,418 → 170,065; independent giảm 378,638 → 207,161 → 206,154. Phần lớn cải thiện xảy ra trước checkpoint đầu; sau đó cải thiện nhỏ hơn. Đây là development curve mô tả, không chứng minh hội tụ hoặc bảo đảm training lâu hơn sẽ không có ích, nhưng chưa cho lý do mạnh để chỉ tăng budget nhằm cứu MARL.

### Cần diễn giải đúng “phối hợp”

Contrast planner PASS so sánh joint lookahead với valuation độc lập cho từng máy rồi **vẫn matching tập trung**. Nó cho thấy xét tương tác tương lai có ích trong solver đã thử; không trực tiếp đo một hệ thống hoàn toàn không có phối hợp.

Independent local và cooperative local dùng cùng thông tin actor cục bộ và cùng giao thức reservation tuần tự, với thứ tự máy luân phiên. Independent có objective/critic từng máy; cooperative có team return, central critic và joint-sequence PPO ratio. Do nhiều thành phần thay đổi cùng lúc, không thể quy toàn bộ 14,25% cho critic hay cho truyền thông riêng lẻ.

Giao thức reservation bảo đảm mọi controller phát ra matching hợp lệ. Collision rejection bằng zero là kết quả của contract này; không chứng minh các agent đã tự học tránh tranh chấp trong một hệ thống bất đồng bộ không có giao thức.

### Không có lợi ích rõ từ matching head linh hoạt hoặc thêm thông tin

Central full matching so với central fixed allocation: delta +0,110, CI 95% [−0,924; +1,143], thắng 5/10 seed. Cooperative full so với cooperative local: delta +0,200, CI 95% [−0,391; +0,790]. Các CI cắt zero; chưa chứng minh matching linh hoạt hay thêm global actor information tạo lợi ích trên điều kiện chính. Cũng chưa phải bằng chứng equivalence, vì chưa có equivalence margin/test.

## 4. Cơ chế giảm cost và waiting

Cooperative local so với independent giảm downtime cost từ 146,488 xuống 116,170, khoảng −30,318; trong khi failure cost **tăng** từ 39,757 lên 40,631 và maintenance cost tăng nhẹ. Failures tăng từ 2,650 lên 2,709. Vì vậy câu mô tả đúng là cải thiện tổng cost chủ yếu nhờ giảm downtime, không phải giảm số failure.

Central fixed so với rule cũng đánh đổi: downtime cost 115,603 so với 124,472, nhưng failure cost 32,916 so với 29,250. Joint planner có failure cost 25,805 và downtime cost 122,325. Các solver đạt tổng cost tương tự với phân bố tổn thất khác nhau; nếu triển khai coi failure nghiêm trọng hơn hệ số 15 hiện tại, thứ hạng cần đánh giá lại theo mục tiêu đã chốt.

| Controller | Request bị censored ở cuối H12 | Mean wait request đã bắt đầu service | P95 wait đã bắt đầu service | Max wait đã bắt đầu service |
|---|---:|---:|---:|---:|
| Rule mạnh | 16,43% | 2,103 | 4 | 6 |
| Joint planner | 19,35% | 1,456 | 4 | 4 |
| Central matching | 13,84% | 0,983 | 3 | 5 |
| Central fixed | 12,28% | 0,970 | 3 | 5 |
| Independent local | 16,31% | 1,082 | 3 | 8 |
| Cooperative local | 11,85% | 0,802 | 4 | 6 |
| Cooperative full | 11,50% | 0,746 | 3 | 6 |

Percentile ở bảng này là nearest-rank trên request đã **bắt đầu** service; chưa chắc repair đã hoàn tất. Những request còn chờ cuối episode không có thời gian chờ đầy đủ. Một số failure xuất hiện đúng biên H nên bị censored với observed wait zero. Do đó không dùng bảng này để suy ra phân phối wait không điều kiện, không kết luận planner starvation-free dù không ghi nhận overdue trên primary panel.

Số request/episode và số terminal pending cũng khác nhau giữa policy; chỉ so censored fraction chưa đủ để đánh giá chất lượng. Tỷ lệ episode violation, request overdue và terminal pending cần báo cáo riêng.

## 5. Sensitivity: kết luận không đồng nhất giữa condition

| Condition / H | Rule chọn từ development | Joint planner | Central matching | Central fixed | Independent local | Cooperative local | Cooperative full |
|---|---:|---:|---:|---:|---:|---:|---:|
| Homogeneous / 12 | 201,238 | 195,498 | 220,484 | 215,563 | 203,188 | 205,310 | 202,061 |
| Homogeneous / 24 | 397,573 | 387,886 | 433,921 | 432,728 | 401,497 | 407,713 | 400,395 |
| Specialized / 12 | 167,695 | 162,073 | 163,279 | 163,170 | 200,335 | 171,795 | 171,995 |
| Specialized / 24 | 336,681 | 327,939 | 327,257 | 328,219 | 406,657 | 344,961 | 345,424 |
| Nominal / 12 | 166,214 | 166,475 | 166,228 | 166,222 | 166,214 | 166,671 | 166,214 |
| Nominal / 24 | 339,684 | 340,120 | 339,740 | 340,101 | 339,684 | 340,084 | 339,684 |
| Skill mask / 12 | 168,238 | 162,027 | 164,081 | 164,219 | 201,358 | 177,680 | 179,065 |
| Skill mask / 24 | 337,230 | 328,303 | 328,766 | 329,324 | 438,475 | 378,443 | 383,696 |

Rule ở homogeneous là deadline matching; ba condition còn lại là adaptive isolated matching. Không thay baseline theo kết quả test.

- **Specialized H24:** central matching có mean thấp hơn joint planner 0,682, nhưng CI 95% [−1,937; +0,573] cắt zero. Chưa có bằng chứng RL vượt planner; cũng không nên nói planner luôn thắng.
- **Homogeneous:** central RL xấu hơn rule, với episode violation 14,50%/11,69% ở H12 và 15,72%/13,84% ở H24. Cooperative local cũng có cost cao hơn independent local. Lợi ích phối hợp học ở specialized không giữ nguyên trên mọi condition.
- **Nominal K4:** mọi arm gần rule, không có waiting violation. Run mới không còn vi phạm nominal guard như thí nghiệm trước. Do code/panel đã thay đổi, không quy chênh lệch giữa hai run thành effect nhân quả của một sửa đổi riêng lẻ.
- **Skill mask:** tại H24, violation của independent/local cooperative/full cooperative lần lượt 45,19% / 29,45% / 33,98%; central matching 3,17%, central fixed 2,53%, joint planner 0,31%. MARL giảm cost so với independent nhưng chưa xử lý tốt service fairness trong điều kiện này. Skill mask đồng thời giảm số edge hợp lệ và capacity hiệu dụng; không phải phép thử thuần information restriction.

## 6. Chi phí tính toán và truyền thông

Benchmark common-state cho primary có 12 state từ một development trajectory/condition, một warm pass và ba timed repeat/controller. Learned arm gộp 10 model nên có 360 mẫu thời gian; control có 36. Đây là đo implementation trên cùng state, không phải benchmark hệ thống phân tán hoặc đo latency SLA.

| Controller | Mean decision latency | P95 latency | Training wall time cả 10 seed | Logical bytes/episode H12 | Logical rounds/episode |
|---|---:|---:|---:|---:|---:|
| Rule mạnh | 0,417 ms | 0,428 ms | — | 1.378 | 24 |
| Joint planner | 8,037 ms | 20,390 ms | — | 1.389 | 24 |
| Central matching | 1,191 ms | 1,233 ms | 5,16 h | 1.389 | 24 |
| Central fixed | 1,190 ms | 1,237 ms | 5,17 h | 1.391 | 24 |
| Independent local | 2,563 ms | 2,606 ms | 1,11 h | 2.055 | 72 |
| Cooperative local | 2,592 ms | 2,715 ms | 1,32 h | 2.065 | 72 |
| Cooperative full | 2,668 ms | 2,787 ms | 1,13 h | 6.709 | 84 |

Central RL quyết định nhanh hơn joint planner khoảng **6,7 lần** trong microbenchmark này, nhưng chậm hơn rule mạnh khoảng 2,9 lần. Cooperative serial implementation chậm hơn central và có logical message cost lớn hơn. Không suy ra MARL về bản chất luôn chậm hơn: actor local hiện vẫn dùng frontend featurization chung, giao thức serial được chạy cùng process, và central inference có một phần critic computation chẩn đoán.

Training tổng 13,89 giờ cho 50 model; checkpoint/development callback thêm khoảng 0,50 giờ, initial-checkpoint development khoảng 0,12 giờ. Equal physical steps không đồng nghĩa equal wall compute: central matching catalogue đắt hơn serial head trong training. Tổng parameter count và khởi tạo được giữ giống nhau, nhưng các arm dùng các tensor/head khác nhau.

Logical communication dùng giá 4 byte/số, provisioning một lần/episode, cộng telemetry, dispatch và reservation. Không phải số byte packet thực, chưa bao gồm latency mạng, mất gói hoặc distributed orchestration. Planner biết dynamics, RL có chi phí training; chưa có so sánh tuning budget ngang nhau hay giá thành triển khai toàn hệ thống.

## 7. Quyết định nghiên cứu được dữ liệu hỗ trợ

1. **Giữ joint rollout làm quality reference và adaptive isolated + matching làm đối chứng nhanh, không học.** Chúng không phải oracle optimum; vẫn có thể cải thiện planner.
2. **Giữ centralized RL như ứng viên khi cần giảm thời gian quyết định của planner.** Lợi ích hiện thấy là gần chất lượng planner trên specialized và khoảng 6,7× tốc độ trong implementation này; chưa đạt gate cải thiện 5% priced/3% base so với rule. Generalization homogeneous cần xử lý trước claim mạnh về robustness.
3. **Chưa phát triển thêm MARL architecture chỉ dựa vào việc có nhiều máy tranh technician.** Trước hết phải xác định nhu cầu triển khai: quyết định phân tán thực sự, thông tin không thể tập trung, giới hạn network hoặc N lớn. Contract hiện tại tự nhiên cho centralized matching.
4. Nếu tiếp tục MARL, câu hỏi tiếp theo cần là: lợi ích dưới một constraint triển khai cụ thể có bù cost, waiting và communication không? Phải giữ semantics và xử lý assignment hợp lệ tương ứng giữa các solver. Không thay objective hoặc chọn subset test để cứu một kiến trúc.
5. Panel test này đã mở. Nếu thay kiến trúc/hyperparameter dựa vào báo cáo này, kết quả sau trên cùng panel là development/exploratory; claim confirmatory mới cần panel chưa mở và protocol mới. Mọi full experiment mới vẫn phải theo workflow tests → Mac smoke → commit/push → người dùng chạy lab → người dùng rsync.

## 8. Fallacy scan — 11/11 checked

| Loại | Đánh giá trong phân tích này |
|---|---|
| Simpson's paradox | CAUTION: dấu cooperative-vs-independent đổi ở homogeneous; giữ riêng condition/H, không gộp thành một claim toàn cục |
| Ecological fallacy | CAUTION: trung bình controller/profile không bảo đảm mọi máy/request được phục vụ; báo cáo machine/request và terminal state |
| Berkson's paradox | NOTE: benchmark chọn sẵn cấu trúc specialized; không coi sample đại diện mọi nhà máy |
| Collider bias | Không lọc test theo outcome hoặc chỉ giữ episode không failure; feasible selection chỉ trên development, vẫn giữ đủ 50 seed |
| Base-rate neglect | Báo cả failures/request count, episode violation và request overdue; không lẫn mẫu số |
| Regression to the mean | Không dùng cải thiện giữa hai run khác protocol làm effect nhân quả; dùng paired controls cùng run |
| Survivorship bias | CAUTION: wait của served request bỏ censored demand; công khai censored fraction và không claim starvation-free |
| Look-elsewhere effect | Family chính gồm bốn contrast với Bonferroni; 92 contrast khác exploratory, không chọn sensitivity thắng để thay primary |
| Garden of forking paths | Protocol/checkpoint/reference đóng trước test trong code snapshot; không có đăng ký bên ngoài có thể kiểm chứng thời điểm, và không retune trên test |
| Correlation vs causation | Intervention so sánh solver trong simulator; không quy effect cho riêng critic, information hoặc coordination vì nhiều yếu tố đổi cùng lúc |
| Reverse causality | Không phát hiện vấn đề cho comparison solver chạy trước outcome; không diễn giải tương quan utilization/failure như cơ chế nhân quả đã xác lập |

## 9. Tái lập và các file đối soát

Phương pháp: tính lại deterministically từ artifacts; replay physics với source đã khớp Git. Verdict: **PARTIALLY_REPRODUCIBLE** — aggregate statistics và dynamics logs tái lập, chưa rerun stochastic training và chưa xác nhận actor inference trên mọi logged state. Đây là lý do Material Passport giữ ANALYZED.

- `independent_analysis.json`: mọi mean/contrast tính lại, request statistics, latency, training time và selected updates.
- `source_audit.json`: 27 source file khớp commit.
- `secondary_audit.json`: machine reconciliation, development curve và consistency giữa profile.
- `physical_replay_audit.json`: kết quả replay toàn bộ 5.160.960 tick.
- `analyze.py`, `secondary_audit.py`: script đọc artifacts, không ghi đè run gốc.

Không chạy training mới, không SSH lab, không chuyển dữ liệu hoặc push code trong bước phân tích này.
