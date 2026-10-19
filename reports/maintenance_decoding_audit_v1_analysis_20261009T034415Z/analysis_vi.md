# Phân tích full decoding audit — 09/10/2026

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-09
- Verification Status: ANALYZED
- Version Label: decoding_audit_validation_v1
- Source: `lab_results/maintenance_decoding_audit_v1_full_cpu_20261008T135102Z`
- Experiment commit: `09e325f377a0e34bcf220e15e2a247e46ee1a126`
- Original checkpoint commit: `4b1cffb0cc1e2413e27f0e2c3a7730f8d3e3d7a8`
- Phạm vi: tính lại từ dữ liệu thô, đối chiếu nguồn checkpoint và replay toàn bộ common DEVELOPMENT states. Không train lại hoặc rerun full test trên Mac.

## Kết luận để quyết định bước tiếp theo

**Giả thuyết “serial greedy là nguyên nhân chính khiến MARL thua centralized RL” không được hỗ trợ trên điều kiện chính.** Joint-MAP không cải thiện MARL cục bộ và chỉ cải thiện khoảng 0,0244% cho MARL toàn cục, rất xa ngưỡng 2% đã khóa. Cả hai primary contrast FAIL. Chi phí MARL vẫn cao hơn centralized RL khoảng 5,35–5,41% dù thay decoder.

Kết quả này có ích: không có lý do để tiếp tục đầu tư vào exhaustive decoding với kỳ vọng đóng khoảng cách chất lượng. Nó chưa xác định được nguyên nhân còn lại: representation, optimization, credit assignment, training coverage và objective trade-off vẫn chưa được tách bằng intervention.

**Điểm yếu cụ thể nhất cần nghiên cứu tiếp là xử lý các request corrective chờ kéo dài dưới skill-mask.** Khoảng 82% cost gap giữa MARL và central ở H24 là waiting-cost gap. Trong đó gần 99% waiting cost của MARL phát sinh từ các request vẫn chưa bắt đầu service ở cuối episode.

## 1. Toàn vẹn và nguồn dữ liệu

Run COMPLETED, checkout sạch, CPU một thread. Bắt đầu 08/10/2026 21:51:03 và kết thúc 09/10/2026 02:42:01 theo Asia/Taipei; khoảng 4 giờ 51 phút. Không có training/optimizer step mới.

| Kiểm tra | Kết quả |
|---|---|
| Output artifacts | 49 SHA-256 và dung lượng đều khớp; tổng 273.659.477 byte |
| Input provenance | 70 file nguồn khớp bản gốc trên Mac: 40 checkpoint + 27 snapshot + 3 metadata |
| Original manifest | SHA-256 khớp provenance và bản gốc |
| Experiment/inherited source | 31 snapshot file khớp Git commit và code dùng phân tích |
| Closed-loop episodes | 317.440; đủ grid controller/replica/profile/shock/condition/horizon, không trùng khóa |
| Decision intervals | Tổng horizon của episode là 5.713.920, khớp manifest |
| Decoder audits | 960; 24.900 candidate likelihoods có complete support và normalization đúng |
| Latency samples | 8.928; common development states, một warm pass và ba timed repeats |
| Request records | 865.337; wait=end−arrival, served/censored loại trừ nhau; đối soát counts và waiting cost |
| Physical feasibility | Invalid assignments bằng zero; cost components và request/failure/pending totals khớp từng episode |

Tính lại toàn bộ mean, 16 decoder contrast và 128 reference contrast từ `episodes.csv`; delta, wins và CI đều khớp summary. So sánh decoder được pair trên cùng checkpoint seed, profile, condition, horizon và shock. New shocks 171000..171019 không trùng source shock panel; 32 profile có nhãn đã mở ở run trước. Đây là contrast decoder có shock mới conditional trên profile/checkpoint quen thuộc, không phải unseen-profile generalization.

Run này không lưu từng transition trong decisions.csv. Vì vậy số decision intervals được đối soát bằng episode grid và horizon; không tuyên bố replay dynamics toàn bộ 5,7 triệu tick từ log. Physics/cost được kiểm tra bằng conservation ở aggregate episode, source integrity và tests của runner.

### Replay actor độc lập trên development states

Dùng 20 checkpoint cooperative đã xác minh, teacher-force toàn bộ matching qua original `_serial` forward pass, độc lập với scorer MAP được tối ưu. Kết quả: support khớp, **greedy action và MAP action đều khớp trên cả 960 state**. Toàn bộ candidate matching hợp lệ.

Sai số log-likelihood lớn nhất Mac versus Linux là 1,1444×10⁻⁵, vượt diagnostic reference tolerance 10⁻⁵ ở một audit row: local seed 160005, state 28, selected likelihood error bằng zero và minimum logged score khoảng −57,98. Cả scorer MAP gốc chạy trên Mac cũng có maximum difference này. Maximum difference cho selected greedy/MAP likelihood là 1,2517×10⁻⁶. Đây không phải tái lập bit-for-bit; không có action mismatch trên panel replay. Không âm thầm tăng tolerance để gọi mọi likelihood exact-match.

Full test MAP probability-mass error tối đa 1,6263×10⁻⁶, dưới guard 10⁻⁵; maximum numerical selection regret bằng zero. Chi tiết nằm ở `development_actor_replay.json`.

## 2. Hai primary contrast đều FAIL

Điều kiện chính: specialized, N=4, K=2, H12. Delta = MAP cost − greedy cost. Đơn vị độc lập là 10 checkpoint training replica gốc; mỗi replica đã lấy trung bình 32 profile × 20 shock. CI 97,5% hai phía, Bonferroni cho hai primary contrast. Không dùng 6.400 episode làm 6.400 training replica.

| Arm | Greedy cost | MAP cost | Mức giảm | Delta / CI 97,5% | Seed thắng | Gate |
|---|---:|---:|---:|---|---:|---|
| Cooperative local | 173,86594 | 173,86594 | **0%** | 0 / [0; 0] | 0/10 | **FAIL** |
| Cooperative full | 174,01016 | 173,96766 | **0,0244%** | −0,04250 / [−0,15661; +0,07161] | 1/10 | **FAIL** |

Cả hai qua waiting và terminal-pending guard. Thất bại nằm ở độ lớn cải thiện, số seed thắng và CI strict upper<0, không phải engineering lỗi.

Đối soát episode cho thấy:

- Local: cả 6.400 cặp episode giống nhau ở toàn bộ 25 physical/request metrics đã kiểm tra, không chỉ mean cost bằng nhau.
- Full: 6.314 cặp có cost bằng nhau; 50 cặp MAP tốt hơn và 36 cặp kém hơn. Toàn bộ 86 nonzero-cost differences nằm ở seed 160003. Seed đó có mean delta −0,425; chín seed khác có mean delta zero. 6.312/6.400 cặp giống nhau ở toàn bộ 25 physical/request metrics.

CI [0;0] của local phản ánh không có biến thiên delta trên panel quan sát. Không coi nó là chứng minh greedy=MAP trên mọi state, mọi shock hoặc mọi policy. Full chỉ có một seed khác zero nên paired-t normality cần thận trọng; tuy nhiên đây cũng không phải bằng chứng cho practical benefit 2%.

## 3. Chất lượng so với các solver khác trong cùng run mới

| Solver trên specialized H12 | Priced cost ↓ | Episode có quá hạn ↓ |
|---|---:|---:|
| Joint rollout | **164,14688** | **0%** |
| Central fixed allocation | 164,93938 | 0,53125% |
| Central matching | 165,04047 | 0,43750% |
| Rule mạnh | 171,89063 | 6,09375% |
| Cooperative local greedy/MAP | 173,86594 | 4,67188% |
| Cooperative full greedy | 174,01016 | 4,21875% |
| Cooperative full MAP | 173,96766 | 4,18750% |

MAP local cao hơn central matching 8,82547 cost, **5,35%**; CI 95% paired [8,14860; 9,50234], thua 10/10 seed. MAP full cao hơn 8,92719, **5,41%**; CI 95% [8,44606; 9,40832], thua 10/10 seed. So với rule, MAP local/full cao hơn **1,15%/1,21%**. Các reference contrast là exploratory, chưa hiệu chỉnh family nhiều so sánh.

Ở primary, MAP local so với central có failure-cost gap +6,27422, downtime-cost gap +2,07563, waiting-cost gap +0,57188 và maintenance −0,09625. Full có failure-cost gap +6,63984 trên total gap +8,92719. Phần lớn primary gap là failure cost. Không nên chỉ tối ưu waiting mà bỏ qua quyết định preventive/corrective và downtime trade-off.

Không so trực tiếp 173,866 của run mới với 171,795 của run cũ để suy ra policy xấu đi: checkpoint không đổi, shock panel đã đổi. Các reference đều được chạy lại cùng panel mới để so công bằng.

## 4. Sensitivity: không condition nào cho cải thiện mean đạt 2%

| Condition / H | Local greedy → MAP | Giảm cost local | Full greedy → MAP | Giảm cost full |
|---|---|---:|---|---:|
| Homogeneous / 12 | 204,516 → 204,254 | 0,1280% | 202,084 → 201,823 | 0,1294% |
| Homogeneous / 24 | 403,229 → 402,335 | 0,2219% | 398,362 → 397,911 | 0,1132% |
| Specialized / 12 | 173,866 → 173,866 | 0% | 174,010 → 173,968 | 0,0244% |
| Specialized / 24 | 344,312 → 344,211 | 0,0291% | 344,978 → 346,300 | **−0,3832%** |
| Nominal / 12 | 168,346 → 168,003 | 0,2037% | 167,925 → 167,925 | 0% |
| Nominal / 24 | 341,291 → 341,119 | 0,0504% | 340,950 → 340,988 | −0,0113% |
| Skill mask / 12 | 179,959 → 179,953 | 0,0038% | 181,013 → 181,005 | 0,0041% |
| Skill mask / 24 | 379,510 → 379,504 | 0,0016% | 384,361 → 385,539 | **−0,3066%** |

Một số homogeneous delta có CI exploratory 95% dưới zero; effect chỉ khoảng 0,1–0,2%, chưa tạo lợi ích thực dụng đã chốt. MAP full ở specialized H24 và skill-mask H24 có mean cost cao hơn greedy; CI cắt zero, không gọi đây là tác hại thống kê đã xác lập. Nó minh họa rằng joint probability cao hơn không bảo đảm trajectory cost tốt hơn.

Trên common development states, specialized có zero greedy/MAP disagreement cho cả hai arm, 120 state-model rows mỗi arm. Nominal có 20/120 disagreement ở mỗi arm, nhưng cả 40 trường hợp đều **tied joint probability**, không có strict probability gain. Full homogeneous có đúng một trường hợp strict gain. Đây là panel một development profile/condition trên rule trajectory; không đại diện toàn bộ learner trajectories.

## 5. Phát hiện quan trọng về request chờ kéo dài

Skill-mask H24:

| Solver | Total cost | Waiting cost | Episode violation | Terminal pending/episode |
|---|---:|---:|---:|---:|
| Central matching | 327,93281 | 0,41438 | 2,95313% | 0,35719 |
| Central fixed | 328,49797 | 0,34500 | 2,45313% | 0,32922 |
| Joint planner | 330,90938 | 0,03750 | 0,31250% | 0,47969 |
| Rule mạnh | 341,30156 | 3,03750 | 17,34375% | 0,47813 |
| Local MAP | 379,50375 | **42,48000** | **29,12500%** | 0,50203 |
| Full MAP | 385,53938 | **47,66625** | **33,28125%** | 0,51469 |

Local gap đến central là 51,57094; waiting gap là 42,06563, chiếm **81,57%**. Full gap 57,60656; waiting gap 47,25188, chiếm **82,03%**. Đây là exact cost decomposition, không phải chứng minh nguyên nhân optimization cụ thể.

Đọc toàn bộ request records, tính lại overdue ticks bằng `max(observed_wait−4,0)` cho cả served và censored request; tổng khớp mọi episode-group waiting cost. Với local MAP: 22.524/22.656 overdue ticks thuộc censored request, **99,42%**. Với full MAP: 25.124/25.422, **98,83%**. Censored ở đây nghĩa request chưa bắt đầu service vào cuối H24; không biết eventual service time sau horizon.

Served-wait mean của local/full MAP chỉ 0,688/0,596 tick, thấp hơn central 1,015; served P95 đều 3. Những con số có vẻ tốt này bỏ qua các request chưa được phục vụ và không thể dùng để kết luận MARL có fairness tốt. Request censor fraction lần lượt 12,39%/12,97%, so với central 8,05%; observed-overdue request counts là 1.886/2.192 so với central 221, trên số episode tương ứng.

Mọi outcome request được giữ trong báo cáo, không loại request cuối horizon. Zero-overdue của planner ở primary không bảo đảm starvation-free: vẫn có censored demand. Số failure thấp hơn trong skill-mask cũng chưa đủ chứng minh tốt hơn, vì một máy vẫn failed và chờ service không tạo thêm failure mới trong simulator.

## 6. Latency và giới hạn triển khai

Common-state mean trên specialized, CPU một thread:

| Solver | Mean latency |
|---|---:|
| Rule | 0,453 ms |
| Central matching | 1,174 ms |
| Central fixed | 1,176 ms |
| Local greedy → MAP | 2,533 → 2,698 ms |
| Full greedy → MAP | 2,629 → 2,790 ms |
| Joint planner | 7,470 ms |

MAP thêm khoảng **6,51%/6,12%** mean latency ở primary. Ở nominal, local/full MAP khoảng 3,916/4,033 ms, so với greedy 2,664/2,742 ms. Không có lợi ích quality đủ rõ để chọn exhaustive MAP làm decoder mặc định hiện tại.

Benchmark chỉ có 12 state/condition từ một rule trajectory, ba repeat và mười model/arm; không phải SLA estimate trên mọi production state. Local/serial còn shared frontend; central giữ phần diagnostic critic như runner gốc. MAP đòi hỏi aggregation/scoring chung, chưa đo distributed communication. Không gọi MAP local là execution phân tán vì actor có local feature contract.

H24 specialized/skill-mask trong run mới có mean centralized RL thấp hơn joint rollout planner. Vì vậy không kết luận planner luôn tốt nhất hoặc tối ưu toàn cục; chất lượng phụ thuộc condition, horizon và objective.

## 7. Ý nghĩa cho hướng MARL

1. **Giữ legacy greedy làm execution baseline, giữ MAP như diagnostic.** Không thêm beam search/exhaustive decoder với kỳ vọng giải quyết khoảng cách đang thấy.
2. **Định vị nút thắt learned policy trước khi thêm architecture.** Full-information cooperative policy vẫn kém central, nên thiếu local observation chưa giải thích toàn bộ gap. Chưa thể quy riêng cho critic, credit assignment hay entropy vì các yếu tố đó chưa được isolate.
3. **Ưu tiên ablation về training coverage và cách học shared-resource decisions.** Giữ physics, reward/cost và decoder cố định; thay một yếu tố mỗi lần. Training hiện chỉ specialized/nominal; skill-mask chưa nằm trong mixture. So cả central và cooperative dưới mixture bao phủ mới, để tách distribution shift khỏi yếu điểm thuật toán. Sau đó mới isolate critic/advantage hoặc factorized policy representation nếu gap vẫn còn. Waiting cost đã có trong objective; vấn đề không phải chương trình quên tính penalty.
4. **Communication experiment vẫn là cổng cho lý do triển khai MARL**, nhưng phải có constraint có căn cứ hoặc gắn nhãn synthetic sensitivity. Dùng local rule/auction mạnh, central predictor với stale observations và same-budget MARL; tính reservation/acknowledgement. Không tạo restriction chỉ để một solver thắng.
5. Nếu nghiên cứu thuật toán MARL mới, đóng góp cần gắn với vấn đề đã đo — giữ các request urgency cao không bị chờ kéo dài, học quyết định chia sẻ technician dưới compatibility constraints, hoặc đạt chất lượng với budget communication cụ thể. Chưa có bằng chứng để chọn một module attention/message/credit-assignment cụ thể ngay bây giờ.

Panel mới này đã mở. Bất kỳ tuning dựa vào phân tích này không được tái dùng nó như sealed confirmatory test; cần protocol/seed panel mới và workflow repository cho experiment kế tiếp.

## 8. Fallacy scan — 11/11 checked

| Loại | Đánh giá |
|---|---|
| Simpson's paradox | Không gộp bốn condition/hai horizon; MAP full có dấu mean đổi ở H24 |
| Ecological fallacy | Không suy từ mean cost hoặc served mean wait sang fairness mọi máy; phân tích request censored |
| Berkson's paradox | Checkpoint đã được development-selected; inference conditional trên bộ checkpoint này, không đại diện mọi thuật toán MARL |
| Collider bias | Không lọc test theo cost/waiting hoặc chỉ giữ successful request; đủ seed và episode |
| Base-rate neglect | Episode violation và request-overdue dùng mẫu số khác; báo failure/request counts và censored fraction |
| Regression to mean | Không gọi absolute cost khác run trước là policy thay đổi; weights giữ nguyên, shock mới |
| Survivorship bias | CAUTION: served mean thấp che waiting cost của unserved requests; công khai censoring và exact overdue decomposition |
| Look-elsewhere effect | Hai primary contrast Bonferroni; sensitivity/reference contrasts exploratory, không lấy homogeneous thắng để cứu primary |
| Garden of forking paths | Decoder/checkpoint/budget/gates đóng trước full run trong committed protocol; analysis bổ sung censoring decomposition mang tính descriptive |
| Correlation vs causation | Decoder intervention cho kết luận hẹp về decoder; cost decomposition không chứng minh credit-assignment/representation là nguyên nhân |
| Reverse causality | Không suy “ít failure gây chất lượng tốt hơn”; failed machine chờ lâu có thể không phát sinh thêm failure |

## 9. Files và tái lập

`independent_analysis.json`: mọi mean, decoder CI, raw paired episode counts, 86 primary nonzero differences, request/censoring decomposition và common-state latency. `development_actor_replay.json`: toàn bộ replay support/action/likelihood, gồm sai số numerical được công khai. `grid_audit.json`: đối soát độc lập đủ 317.440 tổ hợp controller/model/profile/shock/horizon đã khai báo, không thiếu hoặc thừa. `analyze.py`: read-only script để tái tính, không sửa artifacts gốc.

Verdict: PARTIALLY_REPRODUCIBLE. Aggregate arithmetic/statistics và logged matching support/mass tái lập; development actor actions tái lập toàn bộ. Likelihood không bit-identical giữa Mac/Linux. Chưa rerun full test trajectories hoặc training; giữ Verification Status ANALYZED.
