# Phân tích oracle representation — 03/10/2026

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent.
- Origin Mode: validate.
- Origin Date: 2026-10-03.
- Verification Status: ANALYZED; đọc/tính lại CSV, xác minh witnesses và tái tạo exact-oracle arithmetic; không chạy lại training hay checkpoint evaluation.
- Version Label: oracle_representation_analysis_v1.
- Source: `lab_results/ra_qmix_oracle_representation_full_cpu_20261003T041401Z/`.
- Source commit: `a541b2dd677bbf59244e008498c88122549d2b90`; clean theo manifest.
- Protocol: `ra_qmix_oracle_representation_v1`; đã đọc locked plan tại source commit.

## Nhận định chính

Có bằng chứng xác định kiến trúc có giới hạn, nhưng chưa thể quy thất bại adaptation trước đó cho riêng monotonic mixer. Hai kết luận mạnh nhất có phạm vi khác nhau:

1. **Confirmatory structural endpoint:** Q oracle có rank reversals tại một state cố định. Monotonic factorization không thể biểu diễn chính xác toàn bộ bảng Q tại những state này. Điều đó tự nó không chứng minh không biểu diễn được optimal joint action.
2. **Exploratory phát hiện sau run:** shared feed-forward local policy, observation không có machine identity/role, và deterministic greedy argmax không thể tách action khi hai machines có observation/mask giống nhau. Trong hai cells có service profiles giống nhau, oracle cần actions khác nhau. Đây là giới hạn của execution architecture, có thể chứng minh gap từ initial state bằng exact constrained DP.

Primary empirical endpoint **FAIL**, positive-control gate **FAIL**. Không được dùng phát hiện exploratory để đổi kết quả các gate đã khóa thành PASS. Joint-Q có action regret thấp và cost tốt trên các cells này, nhưng không là decentralized execution baseline tương đương.

## Kiểm tra dữ liệu

Run COMPLETED lúc 14:45 ngày 03/10/2026 giờ Đài Bắc, sau khoảng 2 giờ 32 phút. Đủ 120/120 checkpoint files, 6000/6000 unique evaluation rows, 120 models có final update=10000; 12000 progress rows finite. Có 53600 unique per-state model metrics, đúng (375+381+584) × 4 arms × 10 seeds. Mọi output trong manifest hiện diện. Invalid requests=0, cost reconciliation error=0. Sealed 201–300 đóng theo manifest và seed audit; freshness chỉ được chứng minh đối với local declared manifests.

Đã xác minh mọi stored rank-reversal witness từ bảng Q gốc và kiểm tra split không chia cùng observation sang train/held-out. Tái tạo exact stochastic DP trên ba configs bằng implementation nguồn: toàn bộ q95/q1 khớp CSV với max absolute error=0. Đây là consistency check; không là một oracle implementation độc lập mới hoặc tái chạy training. Tests trước push đã đối chiếu tiny DP với brute-force độc lập.

Model capacity: ba factorized arms 30082 parameters, joint-Q 30039 (−0,143%). Cùng labels, split và update budgets; initialization/minibatch matching theo protocol và engineering audits. Joint-Q và signed mixer dùng central joint selection, trong khi monotonic arms dùng local greedy; inference information/selection contract khác nhau.

## Kết quả các hypotheses đã khóa

### Complete-Q structural restriction

| Cell | Reachable states | States có rank reversal | Tỷ lệ |
|---|---:|---:|---:|
| Nominal | 375 | 122 | 32,53% |
| Shared preference | 381 | 118 | 30,97% |
| Pressure | 584 | 110 | 18,84% |

Bellman residual ≤7,11e-15. Đây là chứng cứ toán học trong finite cells này; không phụ thuộc training convergence. Tỷ lệ là trên toàn reachable state population, không phải tần suất states khi policy chạy.

### Primary empirical contrast

Held-out one-step oracle regret, equal cell weighting và paired training seeds:

- Monotonic no-CF mean regret: 1,1219955.
- Signed no-CF mean regret: 1,0803650.
- Signed − monotonic: **−0,0416305**, CI95% **[−0,0872005; +0,0039396]**.
- Giảm **3,71%**, cần ≥20%; absolute reduction 0,04163, cần ≥0,05.
- Cải thiện ở **6/10** seeds, cần ≥8/10.
- Primary hypothesis: FAIL cả ba điều kiện. CI chứa 0; không suy equivalence hay kết luận hiệu ứng bằng 0.

Positive control joint-Q có mean held-out regret **0,0183842** (đạt ≤0,10), nhưng normalized RMSE **0,1817415** (không đạt ≤0,10). Vì gate yêu cầu cả hai, positive control FAIL; empirical attribution theo protocol là inconclusive, `empirical_attribution_supported=false`. Không nới RMSE threshold sau khi thấy kết quả.

## Performance theo từng cell

| Cell | Monotonic no-CF regret | Monotonic CF regret | Signed regret | Joint-Q regret |
|---|---:|---:|---:|---:|
| Nominal | 0,18556 | 0,12589 | 0,15829 | 0,00962 |
| Shared preference | 1,69223 | 1,71041 | 1,70151 | 0,02276 |
| Pressure | 1,48820 | 1,53600 | 1,38129 | 0,02277 |

Exact expected **undiscounted deployed-policy cost từ initial state**, mean qua 10 training seeds:

| Cell | Oracle gamma=1 | Monotonic no-CF | Monotonic CF | Signed | Joint-Q |
|---|---:|---:|---:|---:|---:|
| Nominal | 4,000 | 9,179 | 7,214 | 4,150 | 4,000 |
| Shared preference | 4,000 | 23,643 | 23,643 | 18,985 | 4,000 |
| Pressure | 43,455 | 56,854 | 58,593 | 64,709 | 44,333 |

Joint-Q đạt undiscounted oracle cost ở 10/10 seeds cho nominal và shared preference; pressure vẫn có mean gap 0,8775. Monotonic no-CF nominal đạt oracle cost ở 6/10 seeds: mean xấu bị chi phối bởi bốn seeds còn lại. Signed pressure có held-out regret thấp hơn monotonic no-CF nhưng deployed cost **cao hơn 7,855**: uniform state-regret và on-policy trajectory cost là hai metrics khác nhau.

Các costs trong bảng là exact expectation từ DP, không phải mean của chỉ 50 stochastic episodes. Rollout means xác nhận cùng bức tranh nhưng có sampling noise: shared preference monotonic 24,70 so với exact 23,6425; pressure joint-Q 43,92 so với exact 44,3325. Không gọi mean rollout thấp hơn oracle expected cost là policy tốt hơn oracle.

Joint-Q train normalized RMSE khoảng 0,039/0,044/0,068 theo ba cells, held-out khoảng 0,192/0,175/0,178. Nó fit training tables tốt hơn nhiều nhưng generalization value calibration chưa đạt control threshold. Signed train RMSE không cải thiện so với monotonic: khoảng 0,339/0,306/0,163, monotonic no-CF 0,334/0,301/0,151. Vì vậy không thể coi signed failure là bằng chứng monotonic restriction không quan trọng; optimization, scalar-local bottleneck và representation khác vẫn có thể ảnh hưởng.

Monotonic local argmax khớp exhaustive action selection 100% trên stored states. Signed chỉ khớp khoảng 75–78% held-out states. Nếu đem signed model execution bằng local greedy như RA-QMIX hiện tại, held-out regret tăng lên 2,148/3,150/3,211. Bỏ abs của mixer không tạo một decentralized replacement đã xác minh.

## Exploratory: giới hạn shared local policy khi hai máy giống nhau

Trong current architecture, cùng shared network + cùng observation + cùng mask + cùng deterministic argmax ⇒ cùng action. Agent không có machine ID/role; mixer không chạy khi local execution. Thêm capacity hoặc CF không phá được đẳng thức này. Kết luận giới hạn dưới đây áp dụng cho current **memoryless deterministic** execution; không áp dụng tự động cho recurrent, randomized, role-conditioned hoặc coordinated policies.

| Cell | States có identical local observations | States cần unequal actions để đạt oracle Q |
|---|---:|---:|
| Nominal | 0 | 0 |
| Shared preference | 18 | 13 |
| Pressure | 16 | 11 |

Ví dụ shared preference, state_id=32, time=2: cả hai machines age=2, chưa failed, hai technicians free, queues rỗng, cùng service row (2,3). Optimal actions là **(1,2)** hoặc **(2,1)**. Best oracle q95=−3,71475; best equal-action q95=−22,24. Bất kỳ equal action nào tại state này mất ít nhất **18,52525** discounted cost-to-go so với oracle continuation. Đây là conditional regret ở state, không phải mean trajectory cost.

Để không nhầm reachable-state conflict với impossibility từ initial state, đã tính thêm exact DP với constraint: khi local observations giống nhau, actions phải bằng nhau; mọi states khác cho phép oracle chọn joint action tự do. Đây là một relaxation của shared local policy class: nó bỏ các ràng buộc consistency giữa các states khác. Vì vậy minimum cost của relaxed class là **lower bound** cho cost tốt nhất mà current shared deterministic memoryless policy có thể đạt.

| Cell | Unrestricted oracle cost, gamma=1 | Lower bound với equal-observation constraint | Gap tối thiểu |
|---|---:|---:|---:|
| Nominal | 4,000 | 4,000 | 0 |
| Shared preference | 4,000 | 5,000 | **1,000** |
| Pressure | 43,455 | 53,683875 | **10,228875** |

Gamma=.95 cho cùng kết luận: gaps tối thiểu 0 / 1,156094 / 10,633127. Hai positive gaps chứng minh current shared local execution không thể đạt centralized oracle từ initial state ở hai cells đối xứng này, kể cả nếu tối ưu hóa hoàn hảo. Đây là giới hạn execution/local information, khác với complete-Q monotonicity certificate.

Không quy toàn bộ shared-preference cost 23,6425 cho symmetry: lower bound chỉ là 5,0 và relaxation có thể lỏng. Các ràng buộc biểu diễn/local consistency khác và fitting cũng có thể đóng góp. Cấu hình adaptation trước có 3 machines với service rows khác nhau ((2,4),(4,2),(3,3)); chứng cứ symmetry trong cells này chưa trực tiếp giải thích nominal regression trước đó.

## Ý nghĩa cho bước tiếp theo

Đã có cơ sở để kiểm tra khả năng phá symmetry/điều phối trong kiến trúc, thay vì chỉ bỏ CF hoặc monotonic constraint. Candidate cụ thể là role/identity hay tie-breaking/assignment information phù hợp execution contract, so sánh với baseline cùng thông tin và capacity; chưa có kết quả chứng minh candidate nào khắc phục được vấn đề. Nếu muốn giữ decentralized execution, centralized joint-Q chỉ là diagnostic ceiling. Cần một protocol mới trước khi thêm role features, đổi policy class hoặc chạy lại adaptation trên fresh seeds.

## Statistical limits và fallacy scan

Overall: structural proofs có phạm vi xác định; empirical attribution CAUTION/inconclusive theo gate. Training seed là đơn vị inference n=10. Không coi 6000 episodes hoặc 53600 state metrics là independent training replicates. Student-t primary CI là supportive; normality chưa xác minh và n nhỏ. Các per-cell/CF/deployed-cost comparisons và symmetry lower-bound analysis là secondary/exploratory, không dùng làm multiple-testing superiority claims. Không đổi endpoints/thresholds, không chọn seed/checkpoint tốt nhất.

Coverage: 11/11 fallacy types checked.

| Loại | Kiểm tra/giới hạn |
|---|---|
| Simpson's paradox | Per-cell reporting; uniform state regret và trajectory cost khác nhau, không nhập nhằng khi trend đảo. |
| Ecological fallacy | Mean cost không chứng minh mọi seeds tốt; không suy benchmark nhỏ sang 3-machine adaptation. |
| Berkson's paradox | Không chọn lọc cells/seeds thắng; benchmark population đã khóa, symmetry analysis nêu phạm vi. |
| Collider bias | Không conditioning primary endpoint vào failures/queue events hậu training. |
| Base rate neglect | Joint-Q nominal/shared đạt zero failures trên rollouts; pressure không zero. Không so raw failure/cost across cells như cùng điều kiện. |
| Regression to mean | Có paired controls/fresh training panel; không dùng improvements so với run trước làm causal contrast. |
| Survivorship bias | Đủ toàn bộ 120 models/6000 rows; không bỏ seeds xấu, không retry chọn survivor. |
| Look-elsewhere effect | Primary FAIL được giữ; secondary tốt không thay thế gate. |
| Garden of forking paths | Locked protocol được đọc từ source commit; symmetry bound đánh dấu posthoc, không giả làm preregistered endpoint. |
| Correlation vs causation | Complete-Q/symmetry certificates là mathematical restrictions trong model class; không từ correlation CSV suy nguyên nhân adaptation trước. |
| Reverse causality | Architecture và fitting trước outcomes; không suy gradient hay failure association là mediation. |

Reproducibility: exact-oracle arithmetic consistency verified, training reproducibility chưa được tái chạy. Giữ raw artifacts nguyên vẹn; derived symmetry certificates/bounds và cell means được lưu trong timestamped directory mới dưới `reports/ra_qmix_oracle_analysis_*/`.
