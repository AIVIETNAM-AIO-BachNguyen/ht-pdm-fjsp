# Phân tích backlog-adaptive v1

## Material Passport
ARS experiment-agent/validate;2026-10-05 Asia/Taipei. Verification Status: ANALYZED. Raw run maintenance_backlog_full_cpu_20261004T173740Z; clean commit3be08ac5550a64603ce88634f1043a3e00c6817f. Independent statistics and shared-implementation replay, not independent simulator reproduction. Không chỉnh raw artifact, không chạy experiment mới.

## Kết luận
Baseline ngưỡng theo backlog A chưa chứng minh đủ thay S theo protocol đã khóa. Ratio2 chọn gain0 nên A trùng H hoàn toàn; ratio4 chọn gain1 nhưng không cải thiện H trên eval cell chính. S có lợi thế makespan trong cell high/heterogeneous, nhưng đảo chiều bất lợi rõ ở high/uniform ratio4. Vì vậy chưa thể chốt một solution S ổn định hoặc suy ra cần MARL.
Counterfactual cũ mạnh hơn sau20shock: một yêu cầu PMm5 ởtick9 tránh cascade và makespan315/failure0 trên cả20shock. Đây vẫn là can thiệp hậu nghiệm trên một instance đã chọn, không phải generalization experiment.

## Audit
Run COMPLETED,24phút53giây, đúng hashes source/repair/production/coupling/protocol. Đủ34,560eval,13,824dev,576config/coordination và140counterfactual riêng; không lẫn với experiment khác. Kiểm tra Cartesian keys không thiếu/trùng, finite metrics, feasibility, work balance, PM/CM partitions, cost decomposition, cấu hình resolved, partial/final byte equality. Tái tính dev minima và bootstrap10,000draws/seed922100/24instances khớp summary toàn bộ24cells.
Replay tất cả1,728episode first-shock được ghi:369,745decision và414,720completion; cả140counterfactual/72,873decision/53,760completion. Max numerical mismatch0, runtime excluded. Source pinned từ gitcommit đi kèm trong pinned/. Full trace replay không tái chạy full34,560episode; scientific audit dùng kết quả đã chuyển về Mac.

## Primary kết quả và gate
Primary cell: high pressure, heterogeneous service,64jobs/máy. Đơn vịCI là24instance means, mỗi instance20shock cố định.

| CM/PM | H makespan | A makespan | S makespan | Mean instance-relative (A-S)/S;CI95% |
|---|---:|---:|---:|---|
|1|305.20|305.20|307.18|-0.64%;[-0.86,-0.40]%|
|2|314.26|314.26|307.12|+2.31%;[+1.17,+3.62]%|
|4|325.79|326.34|308.50|+5.76%;[+2.16,+12.09]%|

Ratio2: A/H314.26, S307.13; paired gap7.13ticks,95%CI[3.60,11.19]. Mean instance-relative A chậm hơnS2.31%,CI[1.17,3.62]%; upper3.62% vượt margin2% nên **mean_adequacy=false**. CI chứa2%: chưa chứng minh noninferiority, cũng chưa đủ để khẳng định gap thực tế chắc chắn lớn hơn2%. S thu hồi2.17% thời gian tương đối vớiA (CI[1.12,3.38]%), khác2.31% do mẫu số khác. S thắng16/tie6/thua2instances. Không đạt recovery5% của protocol cũ nếu đối chiếu descriptively; đây không phải gate mới.
Ratio4 tail guard: mean của relative per-instance p95 gap6.80%,CI[-0.88,18.50]%; upper vượt5% nên **tail_guard=false**. CI rộng và chứa0: không tuyên bố A chắc chắn tệ hơn S về tail. p95 làm ở từnginstance rồi aggregate, không lấy p95 của pooled480episodes. Overall adequacy=false vì cả hai gate không đạt.

## A đã thay đổi gì và vì sao chưa giúp?
Dev chọn ratio1:(offset0,gain0),ratio2:(offset1,gain0),ratio4:(offset4,gain1); H offsets0/1/4; S leads4/6/6. Gain0 đúng bằngH về mọi outcome/trace, chỉ decision runtime khác. Ratio2 không có adaptive rule hoạt động trong policy được chọn nên kết quả này đánh giá family đã tune, không bác bỏ mọi cách thích nghi backlog.
Ratio4 high/heterogeneous/long: A326.34 vsH325.79. Chỉ instance920216 đổi mean makespan (H527.85/A541.10);23instance còn lại giống. A waiting255.66 vsH48.55; cost953.71 vs839.03. Quy tắc gửi sớm tạo queue, không đảm bảo được sửa sớm.
Witness920211/shock921209: H vàA có cùng ages/failed/remaining/assigned/work tại mọi tick; queue/action khác nhau. Makespan đều348,service378ticks,zero failures; waitingH8/A473. Đây là ví dụ trực tiếp: gửi yêu cầu sớm làm tăng waiting metric nhưng không đổi service calendar hoặc sản lượng.
Witness920216/shock921201: H702/113failures, A720/113, S312/0. A gửi m5 cùngtick3 sau khi m4 làm virtual backlog tăng, rồi thêm m1 ởtick4; service/production core bắt đầu khácH từtick14. H/A cùng firstfailuretick10. Backlog rule không tránh failure khởi đầu và tạo waitingA2539 vsH1749; chưa xác định đầy đủ chuỗi nguyên nhân chỉ từ hai metrics này. Trace đầy đủ trong witness_decisions.csv.
Dev tuning target là global8cells perratio. Ratio2 A(offset2,gain1) có primary-dev tốt hơn selectedA(offset1,gain0), nhưng global-dev tệ hơn. Đây là tradeoff đã khóa, không phải lý do retune trên evaluation. A có8candidate vsH4. Ratio4 selectedgain1 global-dev tốt hơnH nhưng eval primary không generalize; không kết luận từ một selected dev score.

## S có tốt ổn định không?
Trong high/heterogeneous/long ratio4 S thắng23/thua1instances; A-Smeanrelative5.76% (CI[2.16,12.09]%). Gap bị ảnh hưởng mạnh bởi920216; mô tả leave-one-out riêng chỉ còn2.82%, không loại case khỏigate/CI chính. H/Afailure chỉ12/480shock ở cell này, nhưng concentratedcascade làmmean/tail xấu; Sfailure26/480shock với failure ít hơn trên mỗi episode xấu. Không dùng frequency một mình thay impact.
S vẫn có cascade tại920211/shock921209: makespan543/70failures, trong khiH/A348/0; firstfailureS ởtick10. S queue luôn0 không có nghĩa failed waiting0, bởi máy có thể hỏng và chờ ngoài FIFO.
Ở high/uniform/long ratio4, H455.42/A459.27/S541.51. A nhanh hơnS10.53% khi normalize theoS (CI[1.65,19.72]%); paired S-A=82.24ticks,CI[27.39,141.33]. Đây là secondary exploratory contrast, không kiểm discovery/multiplicity. S thuaA14/thắng10instances. Không gọi S tốt hơn mọi điều kiện. Row-mean service matched không bảo đảm realized chosen duration giống nhau.

## Counterfactual900215: cả20shock
| Variant | Có áp dụng/n20 | Mean makespan | Mean failures | Thắng/tie/thua Sbaseline |
|---|---:|---:|---:|---:|
|S_baseline|0|567.65|38.35|0/20/0|
|S_force_m4_w0_t8|0|567.65|38.35|0/20/0|
|S_force_m5_w0_t8|0|567.65|38.35|0/20/0|
|S_force_m4_w0_t9|20|425.10|18.00|12/2/6|
|S_force_m5_w0_t9|20|315.00|0.00|20/0/0|
|S_force_m4_w0_t10|20|632.95|49.95|12/1/7|
|S_force_m5_w0_t10|20|567.65|38.35|0/20/0|

Primary intervention m5→worker0/tick9:567.65→315, giảm44.51% của mean baseline, failure38.35→0; tất cả20shock makespan315. Cải thiện cả6shock baseline khônghỏng (319→315) và14shock cascade. Prefix trước intervention giữ nguyên, hazard arrays chung theo absolute clock; không hủy queue/service cũ. S sau intervention tự bảo trì m4 từtick11. M4/tick9 đôi khi cũng cứu được nhưng không nhất quán; m4/tick10 làmmean xấu hơn. Các no-op và kết quả bất lợi giữ nguyên.
Case được chọn vì biết failure trước đó và action biết hậu nghiệm nên chỉ chứng minh một quyết định có thể tránh cascade của case này trong simulator. Không có CIpopulation từn1instance dù20shocks; không biến intervention thành policy biết trước stochasticfailure.

## Problem có thể phát biểu và giới hạn
Bằng chứng hiện tại gợi ý vấn đề quyết định **khi nào bảo trì và đưa máy cho thợ nào để kịp trước khi hỏng, với service capacity hạn chế**. Health threshold và backlog scalar chưa đủ mô tả deadline/slack của các máy cạnh tranh một thợ nhanh; queue admission sớm cũng chưa đảm bảo service start sớm. S có thể tận dụng lịch rảnh nhưng cách greedy/latest-slot hiện tại còn tạo quyết định mong manh. Đây là hypothesis ở benchmark synthetic, không khẳng định mọi thuật toán scheduling đều lỗi hay bắt buộc dùnglearning.
Impact chính ratio2 còn nhỏ khoảng2%, ratio4 phụ thuộc cell/rare cascade vàdurationassumption. CM/PM1/2/4 chưa calibrated, môi trường fixed-route/resumable không phải fullFJSP. Chưa đủ căn cứ nâng thành claim impact thực tế hoặc novelty. Bước nghiên cứu hợp lý tiếp theo là kiểm residual gap với một scheduler đơn giản kiểm feasible service-before-deadline/burst demand và một oracle nhỏ; chưa triển khai experiment mới trong lượt phân tích này.

## Fallacy scan11/11
1Simpson: báo táchcapacity/skill/ratio, không dùng aggregate đảo chiều;2Ecological: CIinstance khôngclaimmọiepisode;3Berkson: CFselectedhindsight,evalpanelfull;4Collider: khôngcontrolpostfailure/service đểsuycausal;5BaseRate: báo480shockfailure vàseverity,khôngclassifierclaim;6Regression-to-mean: CF giữpairedseed/baseline,khônggeneralizeextremecase;7Survivorship: Cartesianđủ,khôngcensor/excludeoutlier;8Look-elsewhere: primary2gatesđãkhóa,secondaryexploratory;9Forkingpaths: protocolhash/seed/tuning/stoppingchecked,LOOchỉdescriptive;10Correlationcausation: canthiệpsimulatorconditional,khôngcausalplantclaim;11Reversecausality: action/statebeforehazards,khônggọiwaitingmetriclàcơchếduynhất. Rủi ro còn lại: n24/vàiinstancetạocascade, fixedshockconditionalCI, globaldevsmalln6 vàdomain chưa calibrated.
