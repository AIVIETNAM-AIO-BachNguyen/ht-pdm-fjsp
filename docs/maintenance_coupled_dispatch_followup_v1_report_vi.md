# Báo cáo follow-up: thời điểm yêu cầu và phân công technician dùng chung

**Ngày:** 18-10-2026 (UTC). **Trạng thái:** hoàn tất audit, protocol, tests, smoke, full run và phân tích. Đây là follow-up được thiết kế **sau khi** biết v1/v2; nó không xác nhận hồi tố kết quả của v2. Không có RL training, communication algorithm hay thay đổi physics/cost.

## 1. Kết luận ngắn

Trên sáu ma trận test mới, coupled rollout **không cải thiện** baseline đơn giản đã chọn bằng development. Với cùng request rule `capacity_aware`, nó tăng tổng cost trung bình **22,15** đơn vị/episode (**+2,45%**) so với risk priority; khoảng tin cậy 98,33% theo bootstrap phân cấp cho chênh lệch cost là **[−6,94; +53,26]**, chỉ 2/6 ma trận có lợi. Nó chậm khoảng **1.580 lần** risk priority ở thời gian tính dispatcher/episode. Đây là kết quả về **phương pháp rollout đã thử**, không phải bằng chứng heuristic gần tối ưu.

Hiệu ứng timing `capacity_aware` so với `threshold` dưới risk priority chỉ **−1,52** cost (**−0,17%**), khoảng **[−6,91; +3,28]**; chưa đạt biên lợi ích 3%. Cả 708 tick test có proposal khác nhau trong H đều **không đổi assignment**, state vật lý hoặc cost trong H. Hiệu ứng timing đo được nằm ở cleanup và terminal liability. Interaction timing × allocation trung bình **−2,18** cost, khoảng **[−17,59; +9,59]**, 3/6 ma trận có lợi; chưa đạt biên 2% hoặc yêu cầu nhất quán.

Theo ba tiêu chí đã đăng ký, A/B/C đều **NOT SUPPORTED**. Sáu cấu hình có chủ đích và 16 shock/config không đủ để tuyên bố một quy luật tổng quát rằng timing hoặc allocation vô ích. Hướng hợp lý tiếp theo, nếu có nghiên cứu khác, là khảo sát **điều kiện mà heuristic đơn giản đã đủ tốt** và xây timing rule thực sự thay đổi thời điểm assignment trong H. Follow-up này kết thúc tại đây, không mở thêm vòng tuning.

## 2. Provenance và audit v1/v2

Nguồn gốc đã đối chiếu bằng manifest, SHA256 snapshot và Git blob của commit tương ứng, seed panels, đầy đủ **12 arm/cấu hình**, replay từng tick, bảo toàn request, skill feasibility, completion, FCFS cleanup và cost accounting. Audit PASS cho v1 **2.304 episode/138.240 tick**, v2 **1.536 episode/92.160 tick**; không phát hiện lỗi implementation buộc sửa kết quả cũ. Toàn bộ **60 dòng** kết quả 12 arm nằm trong `artifacts/maintenance_shared_technician_history_audit_20261018T055849Z/all_arms.csv`; ledger, hashes, seed panels và ví dụ state nằm trong `audit.json`. Không mở sealed panel của nghiên cứu khác ngoài hai run được user chỉ định; các panel v1/v2 vốn đã hoàn tất và được audit theo quyền được cấp.

| Nghiên cứu/cấu hình | threshold/FCFS | capacity/FCFS | threshold/risk | threshold/myopic | threshold/lookahead |
|---|---:|---:|---:|---:|---:|
| v1 scarce_overlap | 644,20 | 644,20 | 644,20 | 662,89 | 650,89 |
| v1 partitioned | 854,53 | 851,86 | 854,53 | 1.030,30 | 855,12 |
| v1 ample | 549,98 | 549,98 | 549,98 | 549,98 | 741,78 |
| v2 overlap_service2 | 981,45 | 974,05 | 981,13 | 1.183,44 | 997,00 |
| v2 balanced_service2 | 870,36 | 839,25 | 870,36 | 1.176,83 | 1.097,31 |

Trong v1 `scarce_overlap`, threshold và capacity/FCFS trùng cost, requests, assignments và các chỉ tiêu khác; cơ chế timing không được kích hoạt theo nghĩa tạo quỹ đạo khác. v2 `balanced_service2` cho giảm **31,11/870,36 = 3,57%** ở capacity/FCFS; kết quả này vẫn là **exploratory**. Bảng audit đầy đủ còn có reactive và cả ba timing rule với risk, myopic, lookahead; ví dụ, myopic thường kém FCFS/risk rõ rệt, nên không được xem như baseline mạnh chỉ vì tên gọi.

Audit lookahead cũ chỉ ra giới hạn dự báo, không phải lỗi sai cost/replay: DP đánh giá từng máy với horizon sáu bước và terminal value bằng 0, rồi ghép các marginal gains vào matching tại tick hiện tại. Nó xét skill/worker đang rảnh khi ghép hiện tại, nhưng forecast tương lai mặc nhiên cho worker sẵn và không giữ pool request hay công việc đã cam kết của các máy khác. Tại v2 `balanced_service2`, threshold/lookahead, seed 93000200, tick 5, có hai máy failed đang pending và cả hai technician rảnh; policy chọn WAIT vì các gain hợp lệ bằng −1. Đánh giá lại marginal gain tại chính state này cho horizon 8 cho gain +6: giá trị WAIT phụ thuộc mạnh horizon/terminal value. Ledger audit ghi nhiều tick WAIT dù có failed pending và worker hợp lệ (v1 tổng 1.624, v2 tổng 3.689 trên các arm lookahead). WAIT có thể hợp lệ theo policy, nhưng các giả định availability, thiếu shadow cost chiếm dụng trong tương lai, truncation và việc proposal không nhất thiết biến thành assignment khiến lookahead cũ không là đối chứng dự báo joint-resource mạnh. Chưa có oracle/bound phù hợp để kết luận FCFS tối ưu.

## 3. Bài toán, protocol và phần triển khai

Môi trường chỉ xét bảo trì. Mỗi máy có tuổi, trạng thái failed và lịch yêu cầu; mỗi technician có tập kỹ năng, service duration theo cặp máy–technician và một công việc không ngắt được với thời gian còn lại. `proposal` thêm yêu cầu vào pool bền vững; yêu cầu `pending` chờ tới khi được `assignment`; assigned job chiếm technician tới `completion`, sau đó máy được phục hồi. Máy healthy vẫn degrade khi pending; máy failed vẫn unavailable và có failed-wait/overdue cost. Không reset episode khi tranh technician và không làm mất nhu cầu chưa phục vụ. Objective là tổng maintenance + unavailable + failure + waiting trong **H=48** tick policy và **tail=12** tick FCFS chung, cộng terminal liability cho backlog/công việc còn lại. Physics/cost giữ nguyên v1/v2: tuổi failure bắt đầu ở 4, xác suất shock 0,45, max age 8, restoration 0,75, PM=1, CM=2, unavailable=6/tick, failure=15, failed-wait limit=4 và price=12. Ma trận mới chỉ đổi skill, K=2/3 và service duration 2–4.

Protocol prospective ở `docs/maintenance_coupled_dispatch_followup_v1_protocol.md`, registry cố định ở `configs/maintenance_coupled_dispatch_followup_v1_seed_registry.json`. Sáu ma trận development và sáu ma trận sealed-test **khác nhau** bao trùm hub, balanced, bottleneck, dense3, sparse3, cross3; raw shock seeds development 94000100–107 và test 94000200–215 tách nhau và tách v1/v2. Không có training seeds. Mỗi cấu hình/shock chạy 10 arm: `threshold`/`capacity_aware` × FCFS, risk priority, myopic, isolated lookahead cũ, coupled rollout. Physical shock tape chung giữa các arm ghép cặp; forecast decision, diagnostic và physical dùng role seeds riêng. Baseline đơn giản được chọn **chỉ từ development** theo cost trung bình đều qua sáu cấu hình × tám shock × hai timing: risk **874,375**, FCFS **875,719**, myopic **1.175,156**. `selection.json` đóng băng risk lúc 06:04:55 UTC; test mở lúc 06:05:35 UTC. FCFS vẫn giữ làm mốc test.

Coupled rollout dùng state hiện tại, pool pending, thời điểm request, công việc busy còn lại, skill và mô hình xác suất/chi phí được phép biết. Nó so tối đa 10 partial matchings hợp lệ gồm WAIT và các action heuristic; mỗi action dự báo bốn kịch bản shock chung độc lập với physical tape. Sau action hiện tại, continuation là risk trong H và FCFS trong tail; mô phỏng cùng dynamics và terminal liability. Chọn mean forecast thấp nhất, ưu tiên baseline khi hòa. Đây là **rollout xấp xỉ**, vì giới hạn candidate, bốn scenarios và một continuation policy; không là joint oracle. Diagnostic dùng bank 16 scenarios **khác** chỉ ở state đủ điều kiện đầu tiên của mỗi episode test có rollout. Exact oracle cho state stochastic H60/K3/queue/committed jobs nằm ngoài ngân sách đã khóa.

Ba primary contrasts, theo thứ tự cost thấp hơn là tốt: A = capacity/coupled − capacity/risk (biên −3%); B = capacity/risk − threshold/risk (−3%); C = (capacity/coupled − capacity/risk) − (threshold/coupled − threshold/risk) (−2%). Mỗi contrast lấy trung bình 16 shock ghép cặp **trong** từng cấu hình rồi trung bình đều sáu cấu hình; bootstrap 10.000 lần lấy mẫu lại cấu hình trước, shock vector trong cấu hình sau, 98,33% CI hiệu chỉnh ba so sánh. Điều kiện SUPPORTED: đạt biên thực dụng, CI upper <0 và có lợi ≥4/6 cấu hình. 95% CI chỉ resample shock được báo riêng, không thay thế uncertainty giữa cấu hình. Nếu CI còn bao phủ biên lợi ích, phân loại INCONCLUSIVE; nếu không thì NOT SUPPORTED.

Phần tái sử dụng: `WaitingEnv`, `transition`, `matching_actions`, các request rules, ba dispatcher đơn giản, isolated lookahead, FCFS cleanup và liability. Phần mới: ma trận/split, role-separated scenario bank, coupled forecast/candidate selection, per-tick trace và diagnostic, integrity/replay gates, hierarchical paired analysis và historical audit script. Không thay đổi luật vật lý hay cost nhằm tìm kết quả dương.

## 4. Engineering gate và full run

Trước full: **47 targeted tests PASS** (matching hợp lệ, pool/committed job, forecast–physical replay trên cùng tape kể cả liability, role seed separation, baseline equivalence, artifact gates), compile/whitespace PASS. Smoke riêng `artifacts/maintenance_coupled_dispatch_followup_v1_smoke_20261018T060347Z/`: **20/20 episode**, **1.200/1.200 tick** replay, integrity PASS, 1,124 giây CPU; chỉ dùng ước lượng compute và kiểm tra cơ chế, không dùng cost smoke chọn cấu hình. Gate ghi ở `docs/maintenance_coupled_dispatch_followup_v1_engineering_gate.md`. Full cố định 480 development + 960 test, 4 decision scenarios, 16 diagnostic scenarios/state, giới hạn 7.200 giây, hoàn thành đúng 1.440 episode sau **141,84 giây**; tqdm và `episodes.partial.csv` cho phép theo dõi tiến độ. Code đã push trước full tại commit `d5738202aa712d044e686d256c8f7cdcffd412fc` trên branch `codex/shared-technician-coupled-followup-v1`.

Full artifact: `artifacts/maintenance_coupled_dispatch_followup_v1_full_cpu_20261018T060451Z/`. `manifest.json` = COMPLETED, 1.440/1.440, `integrity.json` = PASS, 480 development/28.800 tick và 960 test/57.600 tick replay. `episodes.csv` có 1.440 hàng; `decisions.csv` có 57.600 test tick; `forecast_diagnostics.csv` có 192 state. Bảy snapshot hashes khớp manifest; selected baseline có timestamp trước test; seed/config split PASS; tất cả assignment hợp lệ, request conservation và hai cách cộng cost khớp. Artifact nguồn và run cũ giữ nguyên. `analysis.json` trong artifact phân tích mới chạy hậu nghiệm chỉ để giải thích phase/mechanism, không chọn policy hay sửa tiêu chí.

Development activation trước test: proposal khác ở 5/6 ma trận, contention ở 5/6, coupled action khác risk ở 6/6. Tuy nhiên `assignment_diff_ticks=0` giữa hai timing rule dưới risk trong toàn bộ H của development. Điều này đã được lưu trước khi mở test; không loại `cross3` hoặc ma trận nào khác.

## 5. Kết quả sealed test

Tổng cost trung bình 16 shock; mọi **10 arm × sáu ma trận** được trình bày. `C` = capacity_aware, `T` = threshold; `Iso` là isolated lookahead cũ, `Joint` là coupled rollout mới.

| Ma trận | T/FCFS | T/risk | T/myopic | T/Iso | T/Joint | C/FCFS | C/risk | C/myopic | C/Iso | C/Joint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hub | 913,4 | 913,1 | 1.179,0 | 1.045,7 | 894,8 | 911,1 | 905,2 | 1.179,0 | 1.043,1 | 895,1 |
| balanced | 943,5 | 940,8 | 1.209,6 | 1.007,6 | 972,9 | 931,6 | 940,6 | 1.209,6 | 1.003,9 | 954,6 |
| bottleneck | 1.160,2 | 1.168,4 | 1.288,1 | 1.259,2 | 1.245,8 | 1.164,6 | 1.168,4 | 1.288,1 | 1.259,3 | 1.235,0 |
| dense3 | 781,6 | 781,6 | 1.165,6 | 916,4 | 823,8 | 781,6 | 781,6 | 1.165,6 | 916,4 | 826,0 |
| sparse3 | 873,8 | 873,8 | 1.214,2 | 963,8 | 864,6 | 872,8 | 872,8 | 1.213,2 | 963,8 | 862,6 |
| cross3 | 750,6 | 750,6 | 1.172,4 | 919,2 | 772,2 | 750,6 | 750,6 | 1.172,4 | 919,2 | 778,6 |

| Contrast đã khóa | Δ cost | Giảm tương đối | 98,33% CI phân cấp | 95% CI chỉ shock | Ma trận có lợi | Biên thực dụng | Kết luận |
|---|---:|---:|---:|---:|---:|---:|---|
| A: C/Joint − C/risk | +22,15 | −2,45% | [−6,94; +53,26] | [+10,00; +34,28] | 2/6 | −3% | NOT SUPPORTED |
| B: C/risk − T/risk | −1,52 | +0,17% | [−6,91; +3,28] | [−4,42; +1,60] | 3/6 | −3% | NOT SUPPORTED |
| C: interaction | −2,18 | +0,24% | [−17,59; +9,59] | [−10,44; +5,49] | 3/6 | −2% | NOT SUPPORTED |

Config-paired A: hub −10,13; balanced +14,06; bottleneck +66,63; dense3 +44,44; sparse3 −10,19; cross3 +28,06. Config-paired B: hub −7,94; balanced −0,19; bottleneck/dense3/cross3 0; sparse3 −1,00. Config-paired C: hub +8,19; balanced −18,06; bottleneck −10,75; dense3 +2,19; sparse3 −1,00; cross3 +6,38. Paired config standardized effects lần lượt **+0,73**, **−0,48**, **−0,21** (dấu âm có lợi). Không coi 96 episode/shock trong mỗi contrast là 96 cấu hình độc lập.

Trung bình sáu ma trận: threshold FCFS **903,84**, risk **904,70**, myopic **1.204,81**, isolated **1.018,63**, coupled **929,02**; capacity FCFS **902,04**, risk **903,18**, myopic **1.204,65**, isolated **1.017,60**, coupled **925,32**. Risk thắng FCFS rất ít trên development nhưng FCFS thấp hơn risk **1,14** cost trên capacity test; lựa chọn baseline được giữ nguyên. Coupled tốt hơn risk ở hub/sparse3, song thua nhiều ở bottleneck/dense3/cross3. Không đổi baseline sau khi thấy test.

**Phân rã phase/cost.** A tăng +22,15 gồm H **+20,35**, cleanup **−0,17**, liability **+1,96**. Component vật lý: unavailable **+20,56**, waiting **+8,63**, failure **−7,50**, maintenance **−1,50**. Rollout giảm trung bình 0,50 failures nhưng giảm availability 0,0165, tăng 3,46 failed-wait ticks và 0,72 overdue ticks; số completions giảm 0,91. Do đó giảm failures riêng lẻ không bù được thời gian máy unavailable và chờ. B có H cost **0,00** ở mọi ma trận, cleanup **−1,65**, liability **+0,13**. Dưới risk, trong test H có **708 proposal khác** và **708 pending-before khác**, nhưng **0 assignment/state/cost tick khác**; tail có 66 assignment khác, 116 state khác và 100 tick cost khác. Pool request bền vững giải thích vì sao proposal khác có thể ảnh hưởng tail mà không đổi lịch thực thi H. Đây là giới hạn cơ chế timing của hai rule đã dùng trên các cấu hình này, không phải do request bị xóa. C thay đổi theo ma trận và không có lợi nhất quán.

Diagnostic cùng state đầu tiên: ở 96 state cho mỗi timing, rollout chọn khác risk tại 44 threshold và 42 capacity state. Trên các state thay đổi, bank dùng để chọn action ước lượng cải thiện trung bình khoảng **−33,97** cost; bank diagnostic độc lập cho cải thiện trung bình **−11,62** (threshold) và **−13,18** (capacity), với 23/44 và 23/42 state có dấu có lợi. Nó cho thấy chọn action có variance/selection optimism; đây chỉ là một state và một continuation giả định, **không** suy ra policy closed-loop cải thiện. Physical closed-loop test mới là endpoint; tại đó A xấu hơn. Rollout dùng khoảng **0,381 s** dispatcher/episode, risk **0,000241 s**, FCFS **0,000207 s**, isolated **0,00210 s**, myopic **0,000625 s**. Thời gian full toàn bộ 141,84 s CPU lab wall, không phải benchmark hardware độc lập.

## 6. Giới hạn và khuyến nghị có điều kiện

Không có oracle exact hay lower bound cho sáu cấu hình, nên kết quả không chứng minh FCFS/risk gần tối ưu. Rollout bị giới hạn bởi 10 ứng viên, 4 forecast scenarios, continuation risk và stochastic value estimation; tốt hơn ở hub/sparse3 cho thấy có thể còn dư địa trong một số topology, nhưng chưa ổn định và không đạt margin. Test gồm sáu ma trận nhân tạo có cùng failure/cost law, bốn máy, K2/K3 và service 2–4; khả năng khái quát ra đội bảo trì thực tế, distribution khác, horizon dài hoặc cost khác chưa được đo. Availability được tính trong H, cost gồm H+tail+liability; không đồng nhất hai đại lượng này. Không có training replication vì không train policy.

Khuyến nghị hiện tại là **nghiên cứu điều kiện mà heuristic mạnh đã đủ tốt**, đồng thời nếu tiếp tục nghiên cứu timing thì thiết kế/đánh giá rule có thể chuyển proposal thành assignment trong H trên cấu hình có contention. Chỉ tiếp tục một chương trình timing–allocation lớn nếu có phép thử prospectively độc lập cho thấy lợi ích allocation ổn định; kết quả follow-up hiện tại không hỗ trợ quyết định đó. Các phát hiện về tail và diagnostic là giải thích hậu nghiệm, không phải hypothesis mới đã kiểm định.

## 7. Artifact và tái lập

- Historical audit: `artifacts/maintenance_shared_technician_history_audit_20261018T055849Z/` (`audit.json`, `all_arms.csv`). Nguồn v1/v2 gốc được giữ nguyên tại các thư mục mà user chỉ định.
- Smoke: `artifacts/maintenance_coupled_dispatch_followup_v1_smoke_20261018T060347Z/`.
- Full: `artifacts/maintenance_coupled_dispatch_followup_v1_full_cpu_20261018T060451Z/` (`manifest.json`, `integrity.json`, `selection.json`, `episodes.csv`, `decisions.csv`, `summary.json`, diagnostic, source snapshot, progress log cùng tên `.log` ở thư mục `artifacts/`).
- Post-run read-only analysis: `artifacts/maintenance_coupled_dispatch_followup_v1_analysis_20261018T060950Z/analysis.json`; tạo lại bằng `uv run --frozen --no-sync python scripts/analyze_maintenance_coupled_dispatch_followup.py <full-run-dir> <new-output-dir>`. Bản analysis 06:08:35 UTC được giữ nguyên nhưng thay bằng bản 06:09:50 UTC sau khi sửa điều kiện kiểm tra số episode trong **script hậu nghiệm**; dữ liệu, thuật toán, protocol, full run và kết quả không đổi.
- Branch: `codex/shared-technician-coupled-followup-v1`; engineering commit trước full: `d5738202aa712d044e686d256c8f7cdcffd412fc`. Report/analysis commit được ghi vào lịch sử Git sau full.

Từ Mac, có thể kéo artifact bằng lệnh sau (không được chạy trong nghiên cứu này):

```bash
rsync -avhP --partial bachng@100.111.83.52:/home/bachng/ht-pdm-fjsp/artifacts/maintenance_coupled_dispatch_followup_v1_full_cpu_20261018T060451Z/ ./maintenance_coupled_dispatch_followup_v1_full_cpu_20261018T060451Z/
```
