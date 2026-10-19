# Phân tích role × peer-information — 04/10/2026

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent.
- Origin Mode: validate, phân tích artifacts.
- Origin Date: 2026-10-04, Asia/Taipei.
- Verification Status: **ANALYZED**; tính lại CSV, kiểm tra artifacts và lookup witnesses; không tái chạy training/checkpoint inference.
- Version Label: role_context_analysis_v1.
- Source: `lab_results/ra_qmix_role_context_full_cpu_20261003T131518Z/`.
- Source commit: `0710ca83663df6173ae9e2a08cb74a528e03c875`, clean theo manifest.
- Locked protocol: `ra_qmix_role_context_v1`; đã đọc plan từ chính source commit.
- Derived artifacts/script: `reports/ra_qmix_role_context_analysis_20261003T165400Z/`.

## Kết luận

Có hai tầng vấn đề. **Information/execution contract ban đầu có giới hạn thật**: shared deterministic memoryless agent thiếu ID không thể phân vai khi hai đầu vào bằng nhau. Trong pressure, thêm ID vẫn chưa đủ cho một policy tối ưu tại mọi reachable states; thêm cả ID và peer observation làm constraint system khả thi. Nhưng **mạng với ID + peer chưa học được policy phối hợp tốt bằng objective fit toàn bộ Q hiện tại**. Đây không phải một bản sửa đã thành công.

H-role **FAIL**, H-peer **FAIL**, positive-control **FAIL**; `empirical_attribution_supported=false` cho cả hai. Vì vậy chưa kết luận thêm features khắc phục được performance, hay toàn bộ mixer cần thay. Bước tiếp theo có giá trị nhất là kiểm tra khả năng học quyết định/phân vai trên cùng information contract, tách khỏi yêu cầu fit chính xác toàn bộ bảng Q.

## Kiểm tra run và phạm vi

Run COMPLETED lúc 22:54 ngày 03/10/2026 giờ Đài Bắc, khoảng 1 giờ 39 phút. Đủ 150 checkpoint files, 150 model rows, 1.500.000 optimizer updates, 15.000 progress rows, 7.500 unique evaluation rows và 67.000 unique state-model rows. Mọi file khai báo trong manifest hiện diện. Mỗi model có đúng logs tại updates 100,200,…,10000; logs finite, CF=0, invalid requests=0, cost decomposition khớp objective tuyệt đối. Local greedy khớp exhaustive maxima trên toàn bộ stored states của bốn monotonic arms.

Manifest ghi checkpoint reload identical, mixer/initialization/minibatch matching và oracle Bellman/observation audits đều pass. Analysis kiểm tra độc lập counts/keys/logs/cost sums và tái tạo paired arithmetic từ model CSV; không tái chạy các checkpoint audits. Sealed 201–300 đóng; seed freshness chỉ trong local declared registry. Cùng ba finite cells đã được xem trong run trước; fresh training seeds/split không biến chúng thành fresh environments.

Bốn arms dùng cùng monotonic mixer, cùng shared agent shape và **31.170 parameters**, CF=0. Joint-Q có 31.095 parameters và central joint selection, là diagnostic control khác execution contract. Train bằng supervised oracle Q MSE, gamma=.95, không phải online RL/adaptation. Fixed ID là machine index, chưa phải learned/transferable role. Peer features đòi hỏi telemetry khi execution; chi phí/trễ truyền thông chưa được mô hình hóa.

## Structural information feasibility

SAT dưới đây nghĩa là tồn tại một **shared deterministic memoryless lookup policy**, trên augmented inputs, chọn một joint action tối ưu gamma=.95 tại **mọi reachable state**. Không nghĩa là đã fit được neural network hay toàn bộ Q table.

| Input cho shared agent | Nominal | Shared preference | Pressure |
|---|---|---|---|
| Local | SAT | UNSAT | UNSAT |
| Local + ID | SAT | SAT | UNSAT |
| Local + peer | SAT | UNSAT | UNSAT |
| Local + ID + peer | SAT | SAT | SAT |

Đã dựng lại augmented vectors từ saved observations và xác minh tất cả **2.846 state assignments** của 7 SAT witnesses: action feasible và đạt oracle q95 optimum với tolerance 1e-8. Bốn direct same-input conflict certificates được đối chiếu optimal actions trong Q CSV. Pressure role-only UNSAT là kết quả CP-SAT trong run, không được re-solve độc lập ở analysis này.

- Shared preference: ID đủ để làm optimal lookup policy khả thi. Peer-only không phá đối xứng: ở state_id=32, hai local observations bằng nhau nên hai vectors `[self, peer]` cũng bằng nhau. Optimal actions chỉ là `(1,2)` hoặc `(2,1)`.
- Pressure: ID phá same-state symmetry nhưng các constraints giữa nhiều states vẫn mâu thuẫn khi thiếu peer. Với cả ID và peer, một lookup policy phù hợp tất cả states tồn tại.
- Nominal: ngay local-only đã khả thi; không có original-identical-input conflict states. Thêm features không bảo đảm cải thiện optimization hay deployed decisions.

UNSAT của role-only trong pressure **không tự chứng minh** không thể đạt optimal cost từ initial state: policy có thể tránh một số states gây mâu thuẫn. SAT với gamma=.95 cũng không tự là chứng minh optimal undiscounted cost. Lower bounds từ run trước giữ nguyên phạm vi của chúng; không gán lại những bounds ấy cho role-only.

## Locked primary endpoints

Held-out one-step oracle regret, equal weighting shared preference/pressure, paired 10 training seeds:

| Contrast | Mean delta | Giảm tương đối | Seeds cải thiện | CI95% | Kết quả |
|---|---:|---:|---:|---|---|
| ID − base | −0,049884 | 4,54% | 6/10 | [−0,121257; +0,021489] | FAIL |
| ID+peer − ID | −0,069497 | 6,63% | 7/10 | [−0,173509; +0,034515] | FAIL |

H-role cần reduction ≥0,10 và ≥20%, ≥8/10 seeds: cả ba không đạt. Diagnostic mean exact deployed cost còn **tăng 0,411713**, nominal tăng **254,82%**, vượt guardrail 10%.

H-peer đạt absolute reduction ≥0,05 nhưng không đạt ≥15% hay ≥8/10 seeds. Diagnostic cost giảm **1,033589**, song nominal tăng **27,75%** so với ID-only, vượt guardrail. Cả hai CI chứa 0; không suy hiệu ứng bằng 0/equivalence từ đó.

Secondary peer-only − base delta **−0,120449**, CI95% [−0,210344; −0,030555], cải thiện 9/10 seeds, khoảng 10,97% regret. Nominal cost tăng từ 4,625 lên 9,1785 và shared-preference cost không đổi. Đây là supportive secondary signal, chưa qua một locked primary superiority gate; không chọn nó để thay kết quả FAIL. Factorial interaction **+0,050952**, CI95% [−0,061931; +0,163835]: chưa có bằng chứng chắc cho synergy của hai features theo regret.

## Performance và nominal regression

Exact expected **undiscounted deployed-policy cost từ initial state**, mean 10 training seeds; thấp hơn tốt hơn. Đây là exact DP expectation, khác mean 50 sampled episodes/model.

| Policy | Nominal | Shared preference | Pressure |
|---|---:|---:|---:|
| Oracle gamma=1 | 4,000 | 4,000 | 43,455 |
| Base | 4,625 | 23,643 | 71,050 |
| ID | 16,411 | 23,643 | 71,873 |
| Peer | 9,179 | 23,643 | 69,878 |
| ID + peer | 20,964 | 23,643 | 69,806 |
| Joint-Q | 4,100 | 4,000 | 48,008 |

Nominal base đạt oracle cost ở **9/10** seeds; ID đạt 3/10, peer đạt 6/10, ID+peer đạt **0/10**. ID+peer có 8 seeds cost=23,6425 và 2 seeds cost=10,25. Vì vậy regression không chỉ là một outlier. Trong shared preference, cả 40 monotonic models đều cost=23,6425; features chưa cải thiện deployed behavior.

Held-out regret trung bình:

| Policy | Nominal | Shared preference | Pressure |
|---|---:|---:|---:|
| Base | 0,137386 | 0,960001 | 1,235334 |
| ID | 0,416613 | 0,947691 | 1,147876 |
| Peer | 0,305093 | 0,900359 | 1,054078 |
| ID + peer | 0,716498 | 0,923405 | 1,033167 |
| Joint-Q | 0,020465 | 0,022338 | 0,093394 |

Trong pressure, ID+peer giảm held-out regret khoảng 16,36% so với base nhưng cost chỉ giảm khoảng 1,75%. Regret trung bình trên states không trọng số occupancy và trajectory cost đo hai điều khác nhau. Trong nominal, train regret cũng xấu: base 0,179760, ID 0,841804, ID+peer 1,234563. Do đó regression không chỉ là lỗi generalization trên held-out split.

Late training normalized MSE (updates 9100–10000) của monotonic arms vẫn khá gần nhau: nominal khoảng 0,182–0,185; shared preference 0,182–0,186; pressure 0,054–0,057. MSE tương tự không bảo đảm action ordering/deployed cost tương tự. Logs này không đủ chứng minh optimizer hội tụ toàn cục hoặc tách nguyên nhân optimization khỏi objective mismatch.

## Mạng có thực sự phân vai?

Trên states có **original local inputs giống nhau và oracle bắt buộc unequal actions**, tổng số state × seed pairs gồm cả train/held-out:

| Cell | Base optimal | ID optimal | Peer optimal | ID+peer optimal | Joint-Q optimal |
|---|---:|---:|---:|---:|---:|
| Shared preference, 130 pairs | 0/130 | 5/130 | 0/130 | 16/130 | 130/130 |
| Pressure, 110 pairs | 0/110 | 6/110 | 0/110 | 6/110 | 102/110 |

ID+peer có một ít phân vai ở shared preference, nhưng chỉ 12,31% critical pairs đúng; pressure 5,45%. Trong pressure, ID và ID+peer đều chọn unequal actions 11/110 pairs, trong đó chỉ 6 pairs tối ưu. Unequal actions không đồng nghĩa assignment đúng.

Tại **shared-preference state_id=32**, cả 40 monotonic models chọn **(1,1)**, dù oracle cần (1,2)/(2,1). Cả 10 Joint-Q models chọn một optimal unequal pair. Tại **pressure state_id=9**, cả 40 monotonic models chọn **(0,0)**, dù optimal pairs là (1,2)/(2,1). Vì vậy SAT của information contract và learned behavior còn cách nhau rõ rệt ngay tại witness states.

Rollout shared preference của bốn monotonic arms trùng behavior summaries: mean collisions=1, waiting=2, failures=0,62; mean sampled cost=22,26. Cost components: maintenance 3, queue 1, failure 9,30, downtime 8,96. Exact expected cost là 23,6425; khác biệt sampled mean là stochastic noise, không phải lỗi cost reconciliation. ID+peer nominal cũng có mean collisions=1 và failures=0,608 so với base 0,1 và 0,056. Những associations mô tả hậu quả, không là mediation proof.

## Positive control và diễn giải kiến trúc

Joint-Q gate mới yêu cầu **mỗi cell** held-out regret≤0,10 và undiscounted policy gap≤2:

| Cell | Regret | Exact cost gap | Gate |
|---|---:|---:|---|
| Nominal | 0,020465 | 0,100 | PASS |
| Shared preference | 0,022338 | 0,000 | PASS |
| Pressure | 0,093394 | **4,553250** | FAIL |

Pressure có 3/10 Joint-Q seeds cost=57,755; các seeds còn lại 43,455 hoặc 44,3325. Mean regret thấp vẫn che các lỗi quyết định trên trajectory quan trọng. Control overall FAIL, nên attribution empirical theo protocol chưa được hỗ trợ. Gate RMSE cũ vẫn FAIL theo run cũ; ở run mới RMSE chỉ descriptive, không đổi lịch sử gate.

Suy luận hợp lý từ dữ liệu:

1. Giới hạn shared local information là thật trong những cells đã kiểm tra; ID và peer phục vụ hai chức năng khác nhau.
2. Information-complete lookup policy khả thi không bảo đảm fixed neural architecture/current MSE training học được nó.
3. Chứng cứ rank reversals của run trước về **full Q-table representation** vẫn còn dù bổ sung features. Ngay cả khi optimal policy có thể factorize, objective buộc fit các Q cho cả joint actions suboptimal có thể vẫn tạo sức ép không phù hợp. Đây là hypothesis cần test, chưa phải nguyên nhân đã xác định.
4. Chưa đủ cơ sở thay toàn bộ mixer. Cũng chưa đủ cơ sở chỉ thêm ID+peer rồi coi vấn đề đã giải quyết.

Đề xuất experiment tiếp theo: giữ ID+peer và execution contract cố định; kiểm tra shared agent có học được **một assignment nhất quán từ SAT witness** bằng policy supervision trước, rồi so objective MSE toàn-Q với objective ưu tiên quyết định/optimal-action ranking trong một protocol mới. Không independently chọn nhãn tie `(1,2)`/`(2,1)` từng state, vì có thể tự tạo mâu thuẫn giữa identical local inputs. Tách training-fit ceiling trên finite population khỏi held-out generalization; dùng final fixed budget và nominal cost guardrail. Cần control đạt pressure gate trước khi dùng performance difference để quy nguyên nhân cho mixer. Analysis này không triển khai/chạy experiment mới.

## Statistical interpretation và fallacy scan

Inference unit là **10 training seeds**, không phải 7.500 episodes hay 67.000 state rows. Paired Student-t CI supportive, n nhỏ và normality chưa được kiểm chứng. Structural feasibility là exact finite constraints, không là statistical performance test. Các per-cell/critical-state breakdowns và peer-only CI là secondary, không multiple-testing-adjusted superiority claims. Không so improvements với run trước như causal contrast vì split/init/input shape đã thay.

Coverage: **11/11 fallacy types checked**.

| Loại | Cách xử lý |
|---|---|
| Simpson's paradox | Report từng cell, giữ nominal regression bên cạnh diagnostic mean reduction. |
| Ecological fallacy | Mean không đại diện mọi seeds; không suy finite 2-machine cells sang adaptation/scale transfer. |
| Berkson's paradox | Không lọc giữ models/cells thắng; critical subset là mechanism endpoint đã định nghĩa. |
| Collider bias | Primary không conditioning vào collisions/failures hậu training. |
| Base rate neglect | Pressure có failure law khác; không so raw failure counts như cùng difficulty. |
| Regression to mean | Paired controls cùng mới; không lấy run cũ làm control. |
| Survivorship bias | Đủ 150 models; không bỏ/retry seeds xấu. |
| Look-elsewhere effect | Peer-only signal không đổi primary FAIL. |
| Garden of forking paths | Giữ locked practical thresholds/control; action breakdown không sửa gate. |
| Correlation vs causation | SAT/UNSAT có phạm vi model class; chưa xác định mediation của feature → gradient → performance. |
| Reverse causality | Features trước outcomes; không suy cost gây training decisions hoặc gradient association là nguyên nhân. |

Overall empirical status: **CAUTION/inconclusive attribution**. Structural information-feasibility conclusions có phạm vi finite lookup-policy class rõ ràng. Raw artifacts giữ nguyên; derived summaries và analysis script lưu riêng.
