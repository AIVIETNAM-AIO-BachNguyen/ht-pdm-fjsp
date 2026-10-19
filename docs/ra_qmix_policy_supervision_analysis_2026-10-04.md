# Phân tích consistent policy supervision — 04/10/2026

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent, validate.
- Origin Date: 2026-10-04, Asia/Taipei.
- Verification Status: **ANALYZED**, kèm replay quyết định từ 180 shared-agent checkpoints; không retrain/full reproduction.
- Version Label: policy_supervision_analysis_v1.
- Source: `lab_results/ra_qmix_policy_supervision_full_cpu_20261003T181732Z/`.
- Source commit: `8d15799f1b992b61ea337d20139aec81933f1b49`, clean theo manifest.
- Locked protocol: `ra_qmix_policy_supervision_v1`, đã đọc từ implementation/plan ở source commit.
- Derived script/summaries: `reports/ra_qmix_policy_supervision_analysis_20261004T045908Z/`.

## Kết luận

**Shared agent với ID+peer và monotonic mixer hiện tại đủ khả năng học policy phân vai tối ưu trên ba finite cells này.** Khi train trên toàn bộ states, cả policy-only lẫn hybrid MSE+CE đạt 100% oracle-optimal decisions và đúng SAT teacher, ở cả ba cells và cả10 seeds. Pure MSE toàn-Q với cùng thông tin/architecture/budget vẫn thất bại ở nhiều quyết định quan trọng.

H-learn **PASS**, H-objective **PASS**, positive control **PASS**, `empirical_objective_attribution_supported=true`. Đây là kết quả confirmatory theo gate của experiment này; không sửa các FAIL gates từ experiments trước.

Trong phạm vi diagnostic đã khóa, có bằng chứng can thiệp training objective/supervision giúp khắc phục failure của policy learning mà không thay mixer. Chưa chứng minh MSE là nguyên nhân duy nhất, gradient dynamics cụ thể, hoặc RL/adaptation ngoài finite oracle benchmark đã được sửa. Supervision dùng một consistent oracle-optimal assignment dựng trên toàn bộ population; đây là thông tin huấn luyện đặc quyền mà online RL thông thường không sẵn có.

## Dữ liệu và verification

Run COMPLETED từ02:17 đến05:26 ngày04/10/2026 giờ Đài Bắc, khoảng3giờ9phút. Đủ240 checkpoint files,240 model rows,2.400.000 updates,24.000 finite progress rows,12.000 unique evaluation rows và107.200 unique state-model rows. Mỗi model có đúng100 progress checkpoints, final update10000. Mọi output trong manifest hiện diện. Actual minibatch hashes khớp giữa bốn arms trong từng cell/regime/seed; invalid requests0, objective khớp tổng cost components, evaluation seeds đúng97200–97249. Sealed201–300 đóng.

Analysis tính lại107.200 state regrets/optimal flags/teacher agreement từ selected actions và raw oracle Q; xác minh1.340 teacher assignments từ augmented input lookup; kiểm tra observation-equivalent split consistency; tính lại primary gates từ model CSV và paired rows. **Reload/replay tất cả180 shared-agent checkpoints**, tổng80.400 quyết định, khớp stored actions. Dùng batch inference rồi fallback per-state nếu cần để tránh nhầm floating-point ties. Joint-Q checkpoint inference không replay; metrics của nó được đối chiếu arithmetic với stored actions/Q. Không retrain và Đã tái dựng finite oracle bằng implementation nguồn, đối chiếu toàn bộ q95 với CSV và tính lại discounted/undiscounted policy costs của cả240models, khớp với artifacts. Đây là arithmetic consistency check với cùng implementation, không là oracle implementation độc lập.

Teacher file SHA256 khớp checkpoint metadata. Theo runner, teacher policy đạt discounted oracle cost và undiscounted gap0 ở cả3cells. Initialization/mixer unchanged/reload bitwise audits trong manifest pass; analysis không tái dựng toàn bộ training history để independently kiểm chứng các audits đó.

Bốn arms: pure normalized Q-MSE; masked local CE từ consistent SAT teacher; Q-MSE+1.0×CE; central Joint-Q MSE control. Ba shared arms cùng31.170parameters, cùng execution ID+peer, original mixer, CF0. Policy-only mixer không nhận gradient, nên nominal parameter count không phải số parameters thực sự được train trong arm này. Joint-Q31095parameters có central joint selection, không là decentralized replacement tương đương.

## Locked primary results: all-state learning ceiling

Ceiling train trên toàn bộ state-action rows. Các cột mang tên heldout ở regime này chỉ là partition đã được train, không là held-out evaluation.

H-learn yêu cầu mỗi cell mean all-state regret≤0,10 và≥8/10seeds có≥99% oracle-optimal decisions. Policy-only đạt regret0 và100% optimal ở **10/10 seeds trong từng cell**: PASS. Hybrid cũng đạt100%, dù không là arm của H-learn.

H-objective dùng equal weighting shared-preference/pressure và paired training seeds, hybrid−Q:

- Baseline mean all-state regret: **1,10622184**; hybrid: **0**.
- Paired delta: **−1,10622184**, CI95% **[−1,20622213; −1,00622156]**.
- Relative reduction **100%**, cải thiện **10/10seeds**; đạt absolute≥0,10, relative≥20%, wins≥8/10.
- Diagnostic mean exact undiscounted cost delta **−14,822325**; cost không tăng.
- Nominal mean cost giảm **81,55%**, thỏa guardrail tăng tối đa10%.
- H-objective: **PASS**. CI supportive theo locked Student-t protocol, không là p-value superiority test.

Exact expected **undiscounted deployed cost từ initial state**, mean10training seeds, thấp hơn tốt hơn:

| Ceiling policy | Nominal | Shared preference | Pressure |
|---|---:|---:|---:|
| Oracle gamma1 | 4,000 | 4,000 | 43,455 |
| Q-MSE | 21,678 | 23,643 | 53,457 |
| Policy CE | **4,000** | **4,000** | **43,455** |
| Q-MSE+CE | **4,000** | **4,000** | **43,455** |
| Joint-Q | 4,050 | 4,000 | 44,245 |

Policy-only và hybrid đạt exact oracle cost ở **mọi30cell×seed models mỗi arm**, không chỉ mean. Pure Q-MSE nominal đạt oracle ở1/10seeds, shared preference0/10; nominal9seeds còn lại cost23,6425. Pressure MSE có all-state regret0,937749 và cost53,45715, so với teacher43,455.

Joint-Q control đạt gate ở mỗi cell:

| Cell | Mean all-state regret | Exact undiscounted gap | Gate |
|---|---:|---:|---|
| Nominal | 0,005827 | 0,050000 | PASS |
| Shared preference | 0,006050 | 0,000000 | PASS |
| Pressure | 0,040141 | 0,789750 | PASS |

Control pass trong ceiling không phủ nhận positive-control failures ở các protocols trước, vốn có regimes/splits/thresholds khác.

## Phân vai và full-Q calibration

Trên original-identical-local-input states mà oracle bắt buộc unequal joint actions:

| Ceiling critical state×seed pairs | Q-MSE optimal | Policy CE optimal | Hybrid optimal |
|---|---:|---:|---:|
| Shared preference | 14/130 | **130/130** | **130/130** |
| Pressure | 13/110 | **110/110** | **110/110** |

Đây là cải thiện cụ thể về assignment, không chỉ giảm average regret nhờ các states dễ. Rollouts ceiling policy/hybrid có mean collisions=0 và waiting=0 ở cả3cells. Nominal/shared failures=0; pressure vẫn có stochastic failures (mean1,46), dù policy đạt oracle expectation. Không nhầm optimality với zero failures.

Hybrid vẫn chưa fit được chính xác toàn Q table. Trên original train partition (đã được train trong ceiling), mean normalized RMSE Q-MSE→hybrid là nominal **0,3038→0,3173**, shared preference **0,3044→0,3203**, pressure **0,1556→0,1715**. Calibration hơi xấu hơn nhưng decisions/cost tốt hơn rất nhiều. Đây là bằng chứng thực nghiệm rằng full-table value error và optimal action ordering không tương đương.

Policy-only có normalized Q RMSE rất lớn (~32–52 trên partitions ceiling): không là lỗi pipeline. CE chỉ tối ưu relative local logits; mixer không được train, total-Q scale không có calibration meaning. Không dùng arm này như một trained Q critic/drop-in TD replacement. Hybrid giữ Q-loss và train mixer, nhưng normalized RMSE vẫn chưa thấp; online TD compatibility chưa được kiểm tra.

Rank-reversal certificates từ experiment trước vẫn cho thấy monotonic architecture không biểu diễn chính xác toàn bộ oracle Q table. Kết quả mới chứng minh điều đó **không buộc policy phải chọn sai** trong các cells này khi có ID+peer và supervision phù hợp. Execution-information restriction và full-Q representation restriction là hai tầng khác nhau; thêm features giải quyết tầng thông tin, CE/hybrid giúp học decision ordering.

## Split80/20: secondary transductive transfer

Teacher được giải trên toàn population, có dùng optimal-action constraints của held-out states; labels vào gradient chỉ ở train states. Vì thế đây là **transductive fixed-teacher transfer trên known benchmark**, chưa phải independent oracle generalization, fresh-environment hay scale-transfer result.

Mean held-out one-step regret:

| Split policy | Nominal | Shared preference | Pressure |
|---|---:|---:|---:|
| Q-MSE | 0,908450 | 1,402227 | 1,038363 |
| Policy CE | 0,053808 | 0,048118 | 0,062833 |
| Hybrid | 0,027147 | 0,004714 | 0,046897 |
| Joint-Q | 0,013136 | 0,039255 | 0,065871 |

CE/hybrid train-partition regret0 ở mọi model. Held-out optimal fractions CE/hybrid lần lượt: nominal99,35%/99,22%; shared99,48%/99,48%; pressure97,03%/97,20%. Không suy hybrid superior từ các unadjusted per-cell secondary comparisons.

Exact deployment cost:

| Split policy | Nominal | Shared preference | Pressure |
|---|---:|---:|---:|
| Q-MSE | 15,071 | 21,678 | 53,626 |
| Policy CE | **4,000** | **4,000** | **43,455** |
| Hybrid | **5,964** | **4,000** | **43,455** |
| Joint-Q | 4,050 | 4,000 | 44,289 |

Policy-only đạt oracle initial-state cost ở cả30splitmodels nhưng không tối ưu mọi held-out state. Policy có thể tránh các lỗi off-trajectory; optimal initial-state cost không đồng nghĩa all-states correctness.

Hybrid nominal seed**97007** có exact cost23,6425;9seeds khác đạt4. Tại held-out state_id**32**,time2, model chọn **(1,1)** thay teacher **(1,2)**, regret**18,75295**. Ngoài ra có2held-out errors nhỏ hơn tại state97/209. Một lỗi assignment lớn đủ làm trajectory cost tăng mạnh dù mean held-out regret nhỏ; không chỉ nhìn optimal fraction hay uniform regret để chấp nhận policy deployment. Đây là secondary residual risk, không làm primary ceiling PASS thành FAIL.

## Ý nghĩa và bước tiếp theo

Trong finite known cells đã kiểm tra, không cần thay monotonic mixer để có decentralized optimal policy: hybrid cùng architecture đã làm được, kể cả mọi states/allseeds trong ceiling. Vì objectives được can thiệp với same initialization/minibatch/budget/information, bằng chứng mạnh hơn correlation giữa gradients và cost. Tuy nhiên treatment gồm **thêm decision supervision từ oracle-consistent teacher**, không chỉ đổi một optimizer hay loss scale; chưa xác định toàn bộ mechanistic cause của original RL failure.

Hướng hợp lý tiếp theo là kiểm tra training phục vụ action ordering/assignment dưới điều kiện thực tế hơn: teacher không dùng held-out constraints, nhiều teacher/tie assignments hoặc data collection policies, fresh environments, rồi online RL/adaptation với nominal trajectory guardrail. Chưa coi fixed-ID/supervised result là learned-role transfer. Không có full-run mới hay sửa architecture trong analysis này.

## Statistical limits và fallacy scan

Inference unit10pairedtraining seeds. 12000episodes và107200state rows không là independent training replicates. Một teacher/cell và một split/cell chưa đo teacher/split variability; seedCI không bao gồm uncertainty ấy. Same known finite environment population; no broad generalization claim. Structural/protocol outcomes giữ nguyên, không lựa chọn cells/checkpoints thắng. Exact DP costs và sampled episode means tách rõ: pressure CE rollout mean41,86 thấp hơn expected oracle43,455 do stochastic sample, không phải beats-oracle claim.

Coverage: **11/11 fallacy types checked**.

| Loại | Kiểm tra/giới hạn |
|---|---|
| Simpson's paradox | Report cells/regimes riêng; nominal split outlier không bị diagnostic mean che. |
| Ecological fallacy | Mean không bảo đảm mỗi seed; ghi rõ seed97007 và phạm vi finite population. |
| Berkson's paradox | Không lọc giữ models/seeds tốt; critical subset là mechanism breakdown. |
| Collider bias | Primary không conditioning vào collisions/failures hậu training. |
| Base rate neglect | Pressure failure law khác; optimal policy vẫn có failures. |
| Regression to mean | Paired contemporaneous controls; không lấy improvement từ run cũ làm causal contrast. |
| Survivorship bias | Đủ240models/checkpoints; không bỏ/retry seeds xấu. |
| Look-elsewhere effect | Chỉ locked ceiling contrast confirmatory; split comparisons secondary. |
| Garden of forking paths | Gates/budget/CEweight đã khóa; không đổi gate để sửa nominal outlier. |
| Correlation vs causation | Controlled objective+supervision intervention hỗ trợ finite-cell attribution, không tự xác định cause của onlineRL failure. |
| Reverse causality | Objectives trước outcomes; cost/failure associations không là gradient mediation proof. |

Overall: locked confirmatory endpoints **PASS**, bounded empirical objective attribution supported. Checkpoint decisions/arithmetic checked; training reproduction chưa chạy. Raw artifacts nguyên vẹn; derived script/checks/means lưu riêng.
