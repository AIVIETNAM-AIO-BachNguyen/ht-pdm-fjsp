# Phân tích maintenance_opportunity_cost_v1 — 10/10/2026

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent.
- Origin Mode: validate.
- Origin Date: 2026-10-10, Asia/Taipei.
- Verification Status: ANALYZED; physics, exact values và aggregate statistics đã tái lập; chưa chạy lại planner inference toàn panel.
- Version Label: opportunity_cost_validation_v1.
- Run: `lab_results/maintenance_opportunity_cost_v1_full_cpu_20261010T074101Z`.
- Commit: `f25876eab7d5ba138a1cfe05d5dc8eb7cb96f736`.
- Phạm vi: phân tích bản đã kéo về Mac, giữ nguyên raw artifacts và protocol.

## Kết quả chính

**Cơ chế opportunity cost tồn tại trong diagnostic nhỏ, nhưng dữ liệu mới chưa hỗ trợ đầu tư ngay vào decoder học assignment linh hoạt.** Joint lookahead cải thiện rõ so với rule đã chọn trên development. Phần cải thiện thêm do mở tập assignment rất nhỏ trong panel hiện tại; hơn nữa, planner không đạt backlog guard đã khóa.

Ba phát hiện quyết định:

1. **H-mechanism có bằng chứng có điều kiện:** với cùng A đang hỏng, chỉ thay age của B có thể làm assignment tốt giữa hai phương án đổi chiều.
2. **H-assignment chưa được xác nhận ở primary:** flexible giảm 0,103% so với fixed; CI 97,5% [−0,075%; +0,277%], toàn bộ dưới ngưỡng đầu tư 1%.
3. **H-headroom có bằng chứng về cost, nhưng chưa qua gate tổng thể:** flexible giảm 4,649% so với rule, CI 97,5% [3,960%; 5,362%]; backlog tăng 0,5625 máy/episode, vượt guard cho phép +0,1.

Kết quả đúng theo protocol là **learning_investment_screen = NOT_ESTABLISHED**. Điều này không bác bỏ centralized RL hoặc chứng minh fixed và flexible tương đương ở mọi setting.

## 1. Tính toàn vẹn và tái lập

Run COMPLETED trên Linux x86_64, CPU, các thread env đều bằng 1; checkout sạch và đúng commit handoff. Thời gian run 30 phút 36 giây, từ 15:41 đến 16:11 ngày 10/10/2026, giờ Đài Bắc.

| Kiểm tra | Kết quả độc lập |
|---|---:|
| Files được manifest hash | 21/21 khớp SHA-256 và bytes |
| Source/runtime files | 7/7 khớp bytes của Git commit; plan cũng khớp |
| Development episodes | 5.376, không trùng episode key |
| Test episodes | 30.720, đủ toàn bộ controller × condition × cohort × shock |
| Logged physical decisions | 368.640, replay toàn bộ đạt |
| Request histories | 74.507, dựng lại từ transition và so từng request đạt |
| Rule actions tái lập | 221.184 quyết định đạt |
| Exact state values | 48/48 tái tính expected Q và giá trị tốt nhất đạt |
| Planned contrasts | 12/12 mean/relative effect/CI khớp summary |

Development selection được tính lại từ development CSV, khớp rule và lambda đã lưu. Không chọn lại baseline bằng test. `episodes.csv` và `episodes.partial.csv` giống bytes.

32 profile development và 64 profile test mỗi condition không trùng cả sau hoán vị machine/technician. Kiểm tra physical signature bỏ restoration trên edge không đủ điều kiện; vẫn không trùng. Raw seed panels không giao nhau và không trùng các registry lịch sử tại commit đã khóa. Không suy rộng thành disjoint với mọi hệ thống vật lý hoặc mọi seed ở các branch khác.

Replay xác nhận initial state, shock/forecast seed domains, state continuity, feasibility, từng thành phần cost, backlog và request censoring. Fixed planner luôn chọn trong tập candidate fixed. Chưa rerun approximate planner forecast scoring cho mọi decision; vì vậy không gọi đây là full end-to-end reproduction của solver.

## 2. Primary contrasts và chất lượng episode

Giữ nguyên N4/K2/H12, tổng cost undiscounted với terminal value bằng zero. Cost càng thấp càng tốt.

| Condition | Rule chọn trên development | Rule cost | Fixed rollout | Flexible rollout | Flexible giảm so rule | Flexible giảm so fixed |
|---|---|---:|---:|---:|---:|---:|
| Homogeneous | adaptive isolated | 185,300 | 178,357 | 178,357 | 3,747% | 0,000% |
| Specialized | deadline matching | 186,556 | 178,504 | 177,902 | 4,639% | 0,337% |
| Skill mask — primary | deadline matching | 190,076 | 181,426 | 181,239 | 4,649% | 0,103% |

Hai contrast chính dùng positive delta = reference cost − flexible cost. Average cùng 32 shock trong từng profile, rồi bootstrap 64 profile với 20.000 draws và analysis seed đã khóa. CI 97,5% cho family hai contrast; uncertainty conditional trên shock panel cố định.

| Primary tại skill_mask | Delta cost | CI 97,5% của delta | Relative improvement và CI | Profile thắng / hòa / thua | Verdict |
|---|---:|---|---|---|---|
| Fixed − flexible | +0,1875 | [−0,1348; +0,5093] | 0,103% [−0,075%; +0,277%] | 35 / 7 / 22 | BELOW_RESEARCH_MARGIN |
| Best rule − flexible | +8,8369 | [+7,6357; +10,0537] | 4,649% [3,960%; 5,362%] | 64 / 0 / 0 | PASS_RESEARCH_MARGIN về cost |

CI assignment bao gồm zero: chưa có bằng chứng flexible tốt hơn fixed ở primary; cũng không chứng minh equivalence. Upper CI dưới 1% cho thấy mức cải thiện lớn hơn research margin chưa được dữ liệu hỗ trợ trong panel này.

Specialized fixed − flexible có CI 95% [0,1323; 1,0698], tương ứng [0,074%; 0,603%]: hiệu ứng nhỏ theo chiều flexible, nhưng **exploratory**, chưa đạt 1%. Homogeneous có cả 2.048 cặp trajectory giống nhau; assignment giữa worker đồng nhất không tạo khác biệt vật lý trong panel đã thử.

Ở skill_mask, fixed đã giữ **97,88% của chênh lệch cost giữa rule và flexible**, tính mô tả bằng (rule−fixed)/(rule−flexible). Đây là phân rã số học trong run, không phải tỷ lệ nhân quả quy cho riêng priority learning.

"Fixed" ở đây không phải một mapping máy→technician bất biến: với mỗi served subset và các resource đang khả dụng, nó chọn assignment có tổng duration/restoration thấp nhất. Nó vẫn được chọn subset và trì hoãn bằng joint lookahead; forecast continuation còn là rule unrestricted chung. Vì vậy kết quả chưa chứng minh policy actor chỉ cần thông tin cục bộ, hoặc rằng mọi tương tác tương lai đã được bỏ đi.

## 3. Diagnostic exact: cơ chế đúng nhưng không tạo root headroom

Ký hiệu theo idea: T1 là technician đa năng, T2 chỉ phục vụ A. Trong code hai người là worker 0 và 1. T1 sửa A trong 2 tick; T2 trong 3 tick; B chỉ được T1 phục vụ. A đang hỏng; so hai root actions cùng phục vụ đúng A và dùng chung exact-optimal continuation.

Ví dụ p=0,25; H=6; age A=3:

| Age B | Expected cost: A→T1 | Expected cost: A→T2, giữ T1 | Tốt hơn trong hai phương án |
|---|---:|---:|---|
| 0 | 30,5469 | 32,7969 | T1 sửa A, giảm 2,25 |
| 1 | 35,5977 | 37,8477 | T1 sửa A, giảm 2,25 |
| 2 | 39,8857 | 39,0000 | T2 sửa A, giảm 0,8857 |
| 3 | 44,6018 | 43,0000 | T2 sửa A, giảm 1,6018 |

Age B=2 chưa phát sinh nguy cơ hỏng ngay tick đầu khi threshold=4, nhưng ảnh hưởng ở các tick tiếp theo đã đổi ranking. Đây là bằng chứng cụ thể về hậu quả tương lai của việc chiếm dụng technician.

Tuy nhiên, **cả 48 state đều có fixed_root_gap bằng 0**. Khi mở quyền chọn served subset, candidate fixed có thể phục vụ A bằng T2 và PM cho B bằng T1 ngay bây giờ, đạt cùng expected value tốt nhất. Ví dụ B age2: action chỉ sửa A bằng T2 và action sửa A bằng T2 + PM B bằng T1 đều có Q=39. Fixed không giữ action thứ nhất cho subset {A}, nhưng giữ action thứ hai cho subset {A,B}.

Vì vậy, sign reversal của so sánh có điều kiện không tự tạo ra nhu cầu học assignment linh hoạt. Kết luận zero gap chỉ áp dụng **root ở 48 state, với continuation unrestricted exact**; không chứng nhận fixed controller globally optimal suốt episode.

Cơ chế còn phụ thuộc horizon: reversal theo age B chỉ xuất hiện ở p=0,25/H6 (hai giá trị age A), không xuất hiện ở H8 trong grid. Hai age A không phải hai replication độc lập; với restoration=1, nhiều Q của chúng trùng nhau. Có 20/48 state mà reserve action tốt hơn fast action, nhưng không dùng tỷ lệ grid được thiết kế để ước lượng tần suất trạng thái tự nhiên.

## 4. Backlog guard thất bại và dấu hiệu ảnh hưởng biên horizon

| Metric skill_mask | Best rule | Fixed rollout | Flexible rollout |
|---|---:|---:|---:|
| Failures/episode | 2,5845 | 2,4116 | 2,4014 |
| Overdue waiting ticks/episode | 0,1382 | 0,0151 | 0,0107 |
| Pending chưa bắt đầu service cuối H | 0,4243 | 0,9956 | 0,9868 |
| Máy còn failed cuối H, kể cả đang CM | 0,9839 | 1,0962 | 1,0781 |
| Technician còn servicing cuối H | 1,3687 | 0,5967 | 0,6094 |
| Request đã bắt đầu service / tổng request | 4.424 / 5.293 | 2.900 / 4.939 | 2.897 / 4.918 |
| Request censored | 869 (16,42%) | 2.039 (41,28%) | 2.021 (41,09%) |
| Mean wait của request đã bắt đầu | 1,2175 | 1,6052 | 1,5920 |
| P95 wait của request đã bắt đầu | 4 | 4 | 4 |

Flexible qua overdue guard: delta −0,1274 tick/episode ≤ +0,25. Nhưng pending delta **+0,5625 > +0,1**, nên service guard tổng thể FAIL. Fixed cũng có cùng vấn đề.

Pending đo CM chưa **bắt đầu** service. Rule có nhiều technician còn servicing cuối H hơn; không coi mọi request đã bắt đầu là sửa xong. Chênh lệch tổng máy còn failed nhỏ hơn chênh lệch pending (flexible 1,0781 so rule 0,9839), dù cùng theo chiều bất lợi cho flexible.

Không chỉ do failure xuất hiện đúng boundary: trong 2.021 censored request của flexible, 365 có observed wait zero; 1.656 đã chờ ít nhất một tick. Với rule, các con số là 442 và 427. Mean observed wait của censored request là 2,045 tick ở flexible và 0,907 ở rule. Không suy ra thời gian chờ cuối cùng hoặc starvation, vì episode kết thúc và populations khác nhau.

Các thành phần cost đều đóng góp vào delta rule − flexible: maintenance +1,5112; failure +2,7466; unavailable +3,0498; waiting +1,5293, tổng +8,8369. Cost giảm không chỉ nhờ waiting penalty, nhưng không phải một cải thiện Pareto về mọi metric phục vụ.

Time profile củng cố quan ngại về biên: flexible pending tăng từ 0,3740 trước tick 8 lên 0,9868 cuối tick 11. Tại tick 11, corrective starts/episode chỉ 0,0239, so với 0,1987 ở rule. Rule giữ pending quanh 0,4–0,5 trong các tick cuối.

Điều này **phù hợp với** mục tiêu hữu hạn H12 và terminal value zero: bắt đầu CM muộn vẫn tốn service cost và unavailable cost; nếu completion không còn tạo lợi ích trong episode và request chưa vượt soft wait threshold, trì hoãn có thể có cost thấp hơn. Dữ liệu này chưa định lượng causal share của terminal treatment; cần một sensitivity được khóa riêng, không sửa objective của run hiện tại để thay verdict.

## 5. Scarcity heuristic và compute

Development chọn **lambda=0 ở cả ba condition**. Scarcity arm vì vậy có cùng policy/cost với isolated arm. Trong family penalty đã thử, positive penalty không cải thiện mean development cost. Không kết luận mọi scarcity heuristic đều vô ích; đây chỉ là một công thức, bốn lambda, panel development và fixed tuning budget đã khóa.

Deadline matching là best fast rule ở specialized/skill_mask. Nó vẫn kém planner khoảng 4,6% về cost, nên các rule đã thử chưa xóa hết headroom cho joint decisions. Tại specialized test, isolated mean thấp hơn selected deadline 0,0015 cost; không dùng điều đó để chọn lại rule sau test.

Latency closed-loop đo trên cùng lab hardware, gồm scoring/matching/validation. State distributions khác nhau giữa controller, nên đây không phải common-state benchmark.

| Skill_mask controller | Mean ms/decision | P50 ms | P95 ms | P99 ms |
|---|---:|---:|---:|---:|
| Best rule | 0,1915 | 0,1835 | 0,2339 | 0,2865 |
| Fixed rollout | 8,8997 | 6,1016 | 34,8111 | 36,2097 |
| Flexible rollout | 10,1398 | 6,0678 | 44,2608 | 45,8989 |

Flexible mất thêm khoảng 13,9% mean latency so fixed để đạt mean cost thấp hơn 0,103%; không có claim chất lượng tốt hơn đã xác nhận ở primary. Trên homogeneous, flexible cùng trajectory/cost với fixed nhưng mean latency cao hơn khoảng 42,8%, do xét thêm assignments tương đương.

Scarcity lambda0 chạy nhanh hơn isolated dù cùng decisions (0,0372 so 0,4726 ms ở skill_mask): implementation scalar enumeration và vectorized reference có overhead khác nhau. Không coi đây là chứng cứ về một cơ chế learning hay kiến trúc mới. N4 enumeration và continuation hiện tại chưa đại diện cho planner tối ưu ở scale lớn; run này chưa có RL latency/training/transfer measurement.

## 6. Diễn giải cho hướng nghiên cứu

**Bằng chứng được hỗ trợ:** trong simulator/panel này, joint decisions tạo headroom về finite-horizon cost; giá trị assignment phụ thuộc trạng thái các máy khác ở diagnostic có điều kiện.

**Chưa được hỗ trợ:** learned opportunity-cost representation, decoder có ngữ cảnh tốt hơn pair scores, full assignment cần thiết, service fairness, frozen size transfer, RL thắng planner, hoặc lợi thế real-time.

Thứ tự hợp lý cho follow-up, nếu tiếp tục hướng này:

1. Kiểm tra độ bền với horizon dài hơn và treatment của cuối episode trên một protocol/panel mới. Giữ nguyên kết quả hiện tại; sensitivity thay objective phải công bố riêng, với terminal semantics dùng chung giữa phương pháp.
2. Khi mục tiêu phục vụ đã rõ, ưu tiên centralized policy học timing/served subset với allocator mạnh, so rule và fixed rollout. Cách biểu diễn nhận N/K thay đổi vẫn có thể đáng nghiên cứu, nhưng transfer cần bằng chứng riêng.
3. Giữ flexible assignment là đối chứng/ablation; đầu tư decoder phức tạp sau khi có headroom vượt phương án đơn giản trong setting đại diện hơn.

Không xem failed gate như bác bỏ RL. Gate nói rằng **combo lý do đầu tư đã đặt trước — thêm assignment >1%, headroom so rule, cơ chế, guard — chưa được đáp ứng**. Test panel đã mở; tuning hoặc thiết kế mới dựa trên các kết quả này cần panel mới để gọi là confirmatory.

## 7. Fallacy scan — 11/11 checked

| Loại | Đánh giá |
|---|---|
| Simpson's paradox | Giữ riêng ba condition; không gộp assignment 0/0,337/0,103% thành claim chung. |
| Ecological fallacy | Mean cost không bảo đảm từng machine/request được phục vụ; backlog và request-level logs đã báo. |
| Berkson's paradox | CAUTION: simulator, profile generator và exact grid được thiết kế; không đại diện mọi hệ thống thực. |
| Collider bias | Không loại episode theo failure/backlog/outcome; first-divergence chỉ là diagnostic mô tả, không dùng để thay primary. |
| Base-rate neglect | Báo served/censored với số request và terminal pending; 20/48 exact-grid cases không phải natural-state prevalence. |
| Regression to mean | Không dùng chênh lệch với run cũ khác protocol để quy kết một thay đổi phương pháp. |
| Survivorship bias | CAUTION: served wait bỏ request còn pending; censored counts/observed wait/boundary arrivals được báo riêng. |
| Look-elsewhere effect | Hai primary dùng CI 97,5%; 10 contrasts còn lại và diagnostics mới là exploratory. Không chọn cell thắng để thay primary. |
| Garden of forking paths | Protocol/source khớp commit trước run; baseline chọn development. Không có external preregistration; post-hoc time/divergence analyses được gắn nhãn mô tả. |
| Correlation vs causation | Within-simulator solver comparison không cô lập mọi cơ chế. Time profile phù hợp boundary effect nhưng không định lượng causal attribution; đổi condition còn đổi capacity. |
| Reverse causality | Controller ra action trước outcome; không suy từ tương quan pending/utilization thành nguyên nhân học opportunity cost. |

## 8. Artifacts phân tích

- `analyze.py`: script read-only đối với run gốc, lưu output vào thư mục report.
- `integrity_audit.json`: checksum, source-to-commit và split audit.
- `physical_replay_audit.json`: toàn bộ dynamics, rule actions và request histories.
- `independent_analysis.json`: mean/contrast/CI, exact values, pooled latency/request metrics và first-divergence counts.
- `time_profiles.csv`: diagnostics theo physical tick, mô tả sau test.

Verdict tái lập: **PARTIALLY_REPRODUCIBLE end-to-end; verified raw aggregates/physics/exact Q**. Không chạy lại toàn bộ stochastic planner inference, không có training trong experiment này.
