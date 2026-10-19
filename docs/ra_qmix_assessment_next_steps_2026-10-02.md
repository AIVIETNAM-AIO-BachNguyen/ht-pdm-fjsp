# Đánh giá RA-QMIX và kế hoạch tiếp theo — 02/10/2026

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent.
- Origin Mode: đánh giá bằng chứng hiện có và plan.
- Origin Date: 2026-10-02.
- Verification Status: đã đọc code hiện tại, code tại commit nguồn, manifests, summaries và CSV; chưa chạy lại training hoặc tái tạo checkpoint outputs.
- Version Label: ra_qmix_assessment_next_steps_v1.
- Phạm vi: passive shared-technician benchmark; các đề xuất dưới đây chưa phải protocol full run đã khóa.

## Nhận định chính

RA-QMIX có tín hiệu cải thiện cost trong benchmark nhỏ, nhưng chưa có bằng chứng đủ mạnh rằng toàn bộ ba thành phần đều cần thiết, lợi ích độc lập với capacity, hoặc recipe thích nghi giữ được performance nominal. Kết quả confirmation ngày 01/10 phải được ưu tiên hơn kết quả screening ngày 30/09: curriculum + replay cải thiện stress ở 10/10 training seeds nhưng không đạt gate nominal. Ưu tiên tiếp theo là attribution và ổn định adaptation; chưa nên mở rộng đồng thời architecture, simulator và quy mô.

## Kiến trúc thực tế

Machine là learning agent; technician là resource thụ động có service time cố định. Mỗi máy chọn defer hoặc yêu cầu một technician; FIFO và action mask thuộc môi trường. Local observation gồm time, age, failed và sáu features/technician. Policy có public resource information khi execution, nên decentralized ở mức quyết định, không phải không cần resource telemetry.

RA-QMIX (`tqmix`) có shared machine network, defer head, technician encoder và query–key edge scores. Khi training, monotonic mixer kết hợp local Q đã chọn với flattened joint observation và congestion context. Execution dùng masked local argmax, không chạy mixer. Trọng số mixer lấy absolute value và ELU tăng đơn điệu, giữ điều kiện monotonic của QMIX ([bài QMIX gốc](https://proceedings.mlr.press/v80/rashid18a.html)).

Các điểm cần phân biệt:

- Edge encoder tạo inductive bias phù hợp technician selection, nhưng trunk vẫn nhận vector technician slots có kích thước/thứ tự cố định. Đây chưa phải kiến trúc bất biến hoán vị hoặc chuyển N/K tự nhiên.
- Queue-conditioned mixer bổ sung availability, busy time và queue length vốn đã có trong joint observation. Đóng góp là cách biểu diễn/conditioning, chưa phải thêm thông tin mới hoặc dự đoán queue tương lai.
- CF loss so sánh local action-value delta với joint delta do chính model tính và detach. Đây là regularizer tự nhất quán; không phải counterfactual rollout của simulator, oracle advantage hoặc causal credit assignment đã được xác minh.
- Monotonic factorization vẫn giới hạn biểu diễn tương tác phụ thuộc joint requests. Chưa có chẩn đoán oracle để quy thất bại cho giới hạn đó.
- Feed-forward QMIX trong repo là baseline adaptation cho observation hiện tại; không nên mặc định là bản tái tạo đầy đủ recurrent QMIX trong bài gốc.

## Bằng chứng kết quả

### Leave-one-out, 5 training seeds, endpoint 50.000 episodes

Nguồn: `lab_results/ra_qmix_leave_one_out_5seed_full_cuda_20260927T095633Z/`. Tất cả train trên nominal, sau đó đánh giá bốn scenarios; stress là chuyển điều kiện, khác thiết kế train riêng từng cell. Evaluation seeds 101–200.

| Variant | Nominal | Early failure | Slow service | Combined pressure |
|---|---:|---:|---:|---:|
| Standard QMIX | 9,500 | 160,252 | 96,567 | 188,347 |
| Full RA-QMIX | 8,100 | 147,167 | 90,753 | 181,185 |
| Bỏ edge | 7,700 | 163,136 | 95,495 | 185,081 |
| Bỏ queue mixer | 8,700 | 149,768 | 85,376 | 168,451 |
| Bỏ CF | 8,400 | 131,884 | 89,900 | 172,806 |

Cost thấp hơn tốt hơn. Không gộp raw costs giữa scenarios có horizon 12 và 18.

Full tốt hơn QMIX trung bình ở cả bốn cell trong panel này; nominal giảm 14,74%. Tuy nhiên bỏ edge tốt hơn Full ở nominal; bỏ queue tốt hơn ở slow/combined; bỏ CF tốt hơn cả ba stress cells. Không thể kết luận tất cả components đều hữu ích. Ví dụ paired delta bỏ queue − Full ở combined là −12,734 (SD giữa seeds 14,645); bỏ CF − Full là −8,379 (SD 5,993). Đây là dữ liệu 5 seeds đã được xem, nên quyết định follow-up cần confirmation độc lập.

Capacity khác đáng kể: QMIX 23.108 tham số; Full 38.210 (+65,35%); no-edge 25.412; no-queue 35.906; no-CF 38.210. No-CF là contrast sạch hơn về capacity vì kiến trúc không đổi. Ablations edge/queue còn thay đổi capacity cùng cấu trúc.

CSV gốc có đủ 20.000 unique episode keys và 50 checkpoints, invalid requests đều bằng 0, cost reconciliation error bằng 0. Riêng nominal/50k, cả năm variants có zero failure events trên toàn panel. Nominal đã bão hòa về failure avoidance; các khác biệt còn lại là maintenance và waiting. Tăng số evaluation seeds đơn thuần có thể không bổ sung thông tin khi policy tránh toàn bộ miền failure.

### Comparison xác nhận cũ, 10 seeds

Nguồn: `passive_technician_confirmatory_full_cuda_20260925T211152Z`. Nominal/50k: Full − QMIX = −0,55, CI lưu trong artifacts [−1,55; 0,45], thắng 7/10 seeds. Slow-service có mean delta −8,9955 nhưng Holm-adjusted p = 0,082. Các contrast Full − QMIX tại 50k không đạt mức 0,05 sau adjustment được báo cáo. Đây là tín hiệu performance, chưa phải superiority ổn định. Không gộp thẳng với leave-one-out vì khác panels/protocols.

### Adaptation: screening tốt nhưng confirmation không đạt

Ngày 30/09, curriculum + replay được chọn: nominal degradation −1,66%, stress-max regret delta −0,6071 trên 5 seeds. Ngày 01/10, cùng recipe khóa trên 10 fresh seeds:

| Criterion | Kết quả | Gate |
|---|---:|---:|
| Mean paired nominal relative degradation | +16,14% | ≤ +5% |
| Seeds có nominal degradation ≤ 10% | 3/10 | ≥ 8/10 |
| Seeds cải thiện stress-max regret | 10/10 | ≥ 8/10 |
| Mean paired stress-max regret delta | −0,2685 | < 0 |

`confirmation_passed=false`. CI95 hỗ trợ cho nominal degradation là [−4,41%; +36,69%], cho stress delta là [−0,4031; −0,1339]. Gate thất bại theo điểm ước lượng và seed count; CI nominal rộng không chứng minh độ lớn suy giảm đã xác định chính xác.

Nominal cost trung bình giữa seeds: control 8,35; treatment 9,45. Tỷ lệ từ hai grand means là +13,17%, khác +16,14% vì primary metric là mean của paired relative ratios từng seed. Không đổi mẫu số khi báo cáo.

Stress-max regret dùng các frozen reference costs, không phải oracle. Delta −0,2685 không có nghĩa raw objective giảm 26,85% ở mọi scenario. Raw mean costs control → treatment: early 137,870 → 86,598; slow 98,506 → 84,588; combined 187,404 → 156,043.

CSV confirmation đủ 12.000 unique rows/60 checkpoints, zero invalid requests, zero cost reconciliation error. Audits trong summary báo matching steps/updates và sealed panel closed; chưa tái thực thi để kiểm chứng mọi audit đó.

Các follow-up khác không nên lặp lại thiếu lý do: uniform domain randomization đã fail gate; fixed nominal mixture cũng fail nominal preservation. Checkpoint 40k được chọn có holdout signal tốt hơn 50k ở 4/5 seeds, nhưng stress checkpoint sweep trên fresh seeds không tái lập giả thuyết queue/CF gây late collapse. Không coi late collapse là cơ chế đã chứng minh.

## Rủi ro mô hình hóa và provenance

Simulator reset age/failed khi service START; technician và machine giữ assignment tới completion, nhưng downtime charge theo failed state, không theo toàn bộ thời gian servicing. Máy đang service tiếp tục age increment ở các bước sau nếu không vừa start; vòng failure không loại máy assigned. Vì vậy đây là benchmark maintenance abstraction, chưa đủ để suy ra production availability hoặc downtime vật lý. Cần timeline tests và báo cáo rõ; đổi recovery-at-completion phải thành experiment riêng, giữ nguyên kết quả cũ.

Training dùng gamma=0,95 trong khi objective báo cáo là tổng cost không discount. Đây là khác biệt mục tiêu cần đánh giá, không tự động là bug. So sánh gamma=1 chỉ trong protocol riêng, áp dụng đồng đều cho baselines.

FIFO xử lý requests đồng thời theo thứ tự máy, có thể tạo ưu tiên index. Failure RNG hiện tiêu thụ draw phụ thuộc trạng thái: cùng episode seed giúp ghép evaluation nhưng không bảo đảm cùng exogenous shocks giữa trajectories. Cần permutation audit và RNG keyed bởi episode/time/machine nếu muốn common-random-number coupling mạnh hơn; phiên bản RNG mới cũng cần ghi rõ.

Working checkout HEAD `abc2f74` cũ hơn code tạo results. Các commits nguồn `abb34b3` (leave-one-out) và `33ab619` (confirmation) tồn tại trong Git local và đã được đọc bằng `git show`; không cần giả định code thất lạc. Working tree có thay đổi scale-transfer và các tài liệu chưa commit: bảo toàn, không gộp vào thay đổi không liên quan.

`stage1_six_method_learning_curves_full_cuda_20260926T103542Z` local chỉ có partial episodes và manifest RUNNING, thiếu summary. Loại khỏi bảng kết quả hoàn tất; chưa kết luận run trên lab còn chạy hay đã dừng vì bản local có thể chưa đồng bộ hết.

## Kế hoạch ưu tiên

### P0 — Chẩn đoán bằng artifacts, chưa train mới

1. Lập bảng registry run/commit/protocol/status/seeds; đối chiếu active branch với commit nguồn; không checkout đè local edits.
2. Với confirmation, so sánh nominal control/treatment tại 240192, 360288, 480384 steps theo từng training seed. Tách maintenance, queue, failures, Q margins và request timing; dùng action traces nếu có, hoặc replay checkpoint đã khóa trên development panel hiện tại.
3. Xác định nominal regression là tăng preventive requests, chọn technician kém, tăng waiting, hay bắt đầu failure. Nếu thiếu trace thì ghi missing, không suy cơ chế từ mean cost.
4. Kiểm tra CF raw/weighted loss và gradient magnitude theo phase, replay shares, state normalization và continuation/exploration. Chỉ sửa khi có bằng chứng wiring hoặc protocol sai.
5. Bổ sung timeline/service-cost audit, permutation tests và exact small-cell oracle gap. Gate là cost sum đúng, mask đúng, timeline đúng với protocol; oracle gap báo cáo, không bắt buộc bằng 0.

### P1 — Tách architecture/capacity và adaptation bằng design nhỏ

Hypotheses đề xuất: (H1) RA-QMIX còn lợi ích so với QMIX được khớp số tham số; (H2) bỏ CF cải thiện nominal preservation khi adaptation mà không mất stress improvement. H2 chỉ là candidate dựa trên ablation, chưa có bằng chứng cơ chế.

Design: 3 model arms × 2 training regimes. Model arms: QMIX khớp tổng tham số với Full (công bố counts agent/mixer và sai lệch còn lại), Full RA-QMIX, RA-QMIX no-CF. Regimes: nominal-only và curriculum + stratified replay đúng recipe confirmation. Không thêm KL, graph model hoặc architecture size-transfer trong vòng này. Giữ original QMIX kết quả cũ làm reference; chỉ thêm arm mới nếu cần contrast cùng panel và khóa từ trước.

Primary adaptation endpoint: paired nominal relative degradation giữa curriculum và nominal-only trong từng architecture, cùng stress-max regret gate. Giữ nominal mean ≤5%, ≥8/10 seeds trong ngưỡng 10%, ≥8/10 stress improvements cho vòng confirmation. Primary architecture contrast: Full − capacity-matched QMIX trong nominal-only; secondary: stress từng scenario, cost decomposition, CF diagnostics. Xử lý family các primary contrasts theo protocol; không chọn endpoint sau khi xem kết quả.

Budget dự kiến: 480384 environment steps/trajectory, cùng optimizer update count và epsilon tính theo steps; checkpoints 240192/360288/480384. 5 paired training seeds cho diagnostic (30 trajectories); chỉ recipe được chọn và control sang confirmation 10 fresh seeds. Nếu workload cao, không giảm seed count sau khi nhìn results; sửa protocol trước khi chạy.

Seed splits đề xuất, chưa khóa: diagnostic train 85000–85004, development eval 85100–85149; confirmation train 85200–85209, fresh eval 85300–85349. Phải quét toàn bộ manifests và source training seed derivation để xác minh chưa dùng/không overlap trước khi chốt. Giữ panel 201–300 đóng; manifest closed không thay thế audit lịch sử.

Stopping: fixed budget; chỉ dừng lỗi, nonfinite hoặc engineering audit fail, lưu FAILED/INCOMPLETE. Không pick checkpoint tốt nhất từ confirmation. Candidate không đạt gate thì ghi thất bại; không mở rộng tune trên chính fresh confirmation panel.

Artifacts: timestamped directory mới, manifest + Git SHA/dirty status + protocol version + resolved configs + parameter counts + training schedules/updates + raw/weighted CF losses + training/evaluation CSV + cost decomposition + decision traces + checkpoints + paired seed summary/uncertainty + audit report. Không điền 0 thay metric thiếu.

### P2 — Robustness vật lý và scale, sau attribution

Thí nghiệm recovery-at-completion/servicing downtime riêng với cùng baseline panel; thêm initial ages lệch nhau và horizon dài để tránh nominal chỉ là deterministic maintenance schedule. Dùng cost/machine/timestep, failure rate, waiting/request, utilization cùng total cost. Giữ semantics cũ làm đối chứng, không âm thầm sửa simulator rồi so sánh với raw costs cũ.

Scale-transfer draft có 5/3 và 8/4 complete/sparse graphs là hướng hợp lý, nhưng Relational Set-QMIX là kiến trúc mới; phải đối chiếu capacity-matched pooled-set và permutation equivariance. Không gọi kết quả nó là zero-shot transfer của checkpoint RA-QMIX fixed-shape hiện tại. Chưa có full scale-transfer result được xác minh trong đánh giá này.

### P3 — Final evaluation

Khóa architecture, recipe, checkpoint rule và semantics trước khi mở fresh test panel; đủ training seeds độc lập, baselines tương xứng capacity/steps, oracle trên small cells và reports per scenario. Claim phù hợp hiện tại: technician-aware value decomposition có tín hiệu performance trên synthetic shared-resource benchmark, nhưng component utility và nominal–stress tradeoff phụ thuộc điều kiện.

Mọi experiment code mới tuân thủ `docs/experiment_workflow.md`: protocol → implement/tests → smoke CPU trên Mac → commit/push → bàn giao compound command cho người dùng chạy lab → người dùng rsync với partial/progress. Đánh giá này không sửa code, không chạy full experiment, không SSH/rsync và không commit thay đổi đang có.
