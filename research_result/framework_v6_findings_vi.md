# V6: parameter sharing của edge head trong RL điều phối bảo trì

Ngày phân tích 2026-10-20 (Asia/Taipei). Method proposal và protocol đã khóa ở `research_result/method_v6_proposal_vi.md` và `research_result/framework_v6_protocol_vi.md`; code/test/smoke được commit và push ở `dbf40141e246c32b451ddf992bd3d762d427c6bb` trước FULL. Raw FULL: `research_result/framework_v6_full_20261019T193747Z/`; independent audit: `research_result/framework_v6_audit_20261020T005927Z/independent_audit.json` **PASS**. Mọi raw/checkpoint nằm trên Ubuntu lab, ngoài Git. Không đổi physics, objective, seeds, actor hay checkpoint selection sau TEST.

## Câu hỏi và phép thử đã khóa

Sau khi age2 anticipation v4 và post-decision queue opportunity v5 không tạo incremental benefit, v6 kiểm tra trực tiếp thành phần **chia sẻ cùng edge scorer giữa các cặp machine–technician**. `shared_edge` có machine/worker encoders dùng chung, edge MLP 40→32→1 dùng chung cho 8 vị trí và logit matching bằng tổng edge score. `unshared_edge` dùng cùng encoders, matching-sum decoder và critic, nhưng có 8 edge MLP 40→4→1 độc lập. Edge heads có 1.345 và 1.352 tham số; tổng policy 9.470 và 9.477. Vì width mỗi head khác nhau để khớp ngân sách, đây là contrast **matched-budget construction** của weight sharing, không phải can thiệp toán học chỉ đổi một biến. `flat` 56→MLP→21 logits (17.046 tham số) là architecture control. Cả ba train MaskablePPO từ đầu, cùng thông tin 56 features, legal mask, 21 partial matchings kể cả WAIT, H48+tail16 cost, N4/K2 và bốn case đã khóa. Mask bảo đảm feasibility; đó không phải năng lực policy tự học.

Năm seed huấn luyện × ba arms ×100.352 PPO steps =1.505.280 steps, common case/physical tapes trong mỗi seed; hai checkpoints 50.176 và 100.352. DEV 16 seed/case chọn một checkpoint **trong mỗi training seed và arm**. TEST mới 64 physical seeds/case cho 5 training seeds, thêm FCFS/risk/16-scenario horizon-6 rollout; 1.920 DEV, 4.608 TEST. Primary là trung bình cân đều bốn case của `shared_edge − unshared_edge` trên 5×64 train-seed/physical-seed cells, thấp hơn tốt hơn. CI 95% hierarchical paired bootstrap 20.000 lần resample training seeds và physical seeds; confirm chỉ nếu mean<0 và upper CI<0. Case-level/other baselines là secondary descriptive, không điều chỉnh multiple comparisons.

## Kết quả xác nhận và giới hạn

**Primary đạt tiêu chuẩn đã khóa, nhưng sát ranh giới:** shared − unshared = **−85,592 cost**, CI95 **[−171,240; −0,064]**. Trong 320 cells, shared thắng 215 và thua 105. Ba training seeds 67000000/01/04 có mean effect −164,70/−111,95/−208,63; hai seeds 67000002/03 có +20,91/+36,40. Vì chỉ có năm training seeds và upper CI rất gần zero, ước lượng lợi ích không vững; không gọi đây là bằng chứng policy tốt nhất hoặc novelty. Case-balanced mean cost: shared **955,03**, unshared **1040,62**, flat **899,08**, risk **890,04**, rollout **849,72**, FCFS **941,17**. Shared còn kém flat/risk/rollout trên panel đầy đủ.

| Case | Flat | Unshared | Shared | Risk | Rollout | Shared − unshared (CI95 descriptive) |
|---|---:|---:|---:|---:|---:|---:|
| Interchangeable scale1 | 771,97 | 830,75 | 728,52 | 738,53 | 683,47 | −102,23 [−154,88; −46,98] |
| Reciprocal scale2 | 813,47 | 1086,73 | 959,53 | 809,61 | 723,58 | −127,19 [−282,31; +23,94] |
| Reciprocal scale3 | 1000,67 | 1208,95 | 1138,95 | 1012,91 | 1021,47 | −70,00 [−203,91; +60,19] |
| Specialist/generalist scale2 | 1010,21 | 1036,06 | 993,12 | 999,11 | 970,34 | −42,95 [−112,23; +7,48] |

Shared − flat = +55,95, CI95 [−23,61; +137,55]; shared − risk = +64,99 [−13,66; +146,87]; shared − rollout = +105,31 [+26,38; +185,32]. Unshared − flat = +141,54 [+114,14; +169,91]. Đây là secondary contrasts, không có claim xác nhận thêm. Trên interchangeable và specialist, shared − flat lần lượt −43,45 [−57,19;−30,06] và −17,10 [−31,52;−3,08], nhưng reciprocal2/3 cho +146,07/+138,28 với CI rộng: thiết kế hiện tại không bảo toàn hiệu quả khi worker speed tradeoff tăng. Rollout cùng current-information contract nhưng compute khoảng 0,653 s selection/episode so shared 0,032 s; đây không phải compute-matched performance comparison. Tổng training wall: flat 417 s, unshared 595 s, shared 485 s trên CPU single-thread.

## Mechanism diagnostics sau TEST

Independent verifier PASS source/artifact/checkpoint hashes, fresh splits/grid, regenerated paired training tapes, common structured encoder/critic starts, DEV-only selection, reward/cost reconciliation, request conservation, zero invalid assignments và tính lại toàn bộ hierarchical bootstrap. Campaign audit PASS. Post hoc replays kiểm tra checkpoint hashes, 48 quyết định/episode và tái hiện đúng objective từng TEST episode; chúng **không** là endpoint cost mới.

- True-state machine/worker relabel replay `research_result/framework_v6_permutation_20261020T005944Z/`: 61.440 states/arm. Shared max absolute remapped logit error 2,38e−7, zero unambiguous action changes; 5.142 near-tie action changes do numerical tie order. Unshared có 31.244 unambiguous changes, mean abs logit error 0,457. Chỉ là N4/K2 actor equivariance, không chứng minh trajectory invariance, size transfer hoặc scalability.
- WAIT replay `research_result/framework_v6_wait_20261020T010501Z/`: 3.840 learned-policy episodes, 184.320 decisions. Shared và flat chọn WAIT **0 lần** khi có service hợp lệ trên cả năm replicas. Unshared voluntary WAIT trên legal-service ticks lần lượt 47,6%, 46,7%, 7,5%, 12,9%, 41,7% theo seed 00–04. Undermaintenance/idle opportunity là một cơ chế phù hợp với unshared cost cao ở 00/01/04, nhưng không giải thích toàn bộ contrast.
- Matching replay `research_result/framework_v6_matching_20261020T010937Z/`: 1.280 selected shared episodes, 61.440 decisions. Không có một action nào là matching có thể mở rộng nghiêm ngặt bởi một legal matching khác; mỗi lúc 2 service legal, actor chọn 2. Seeds 02/03 chỉ khởi động 6.307/6.377 service trong H trên 256 episodes/seed, so 8.675/8.488/8.463 ở 00/01/04. Chỉ seeds 02/03 chọn assignment có tổng service duration lớn hơn min giữa legal assignments phục vụ **cùng tập máy**: 536/516 ticks, tổng chênh duration 3.688/3.522. Ở reciprocal2, H starts/episode 18,41/18,64 so 35–37 ở seeds tốt; reciprocal3 12,30/12,59 so 27–29. Đây là liên hệ cơ chế hợp lý giữa worker assignment chậm, chiếm dụng capacity và cost cao. Chưa có closed-loop intervention nên không gán causal effect cho lựa chọn duration; min duration có thể dùng worker khan hiếm hơn ở tương lai.

DEV của hai weak shared replicas cũng xấu ở cả checkpoints 50k/100k, nên không chỉ là lỗi chọn checkpoint. Mean TEST cost shared theo seed 00–04 là 868,7/888,8/1069,0/1067,8/880,9; flat ổn định hơn quanh 899. Nhóm weak shared có H availability khoảng 0,43 so 0,53–0,54 nhóm tốt. Shared policy có thể học cách gán technician gây thời gian service cao trong case reciprocal; weight sharing một mình không giải quyết credit assignment cho thời gian chiếm dụng nguồn lực.

## Kết luận và hướng tiếp

Trong setting này, **shared edge head có lợi ích được xác nhận theo protocol so matched-budget unshared edge heads**, chủ yếu bằng cách tránh một số hành vi undermaintenance của unshared. Lợi ích không đồng đều qua training seeds/cases và không vượt flat/risk/rollout pooled. Shared edge scoring/permutation bias đã có trong literature, đặc biệt *Learning to Dispatch* (NeurIPS 2020) và HetGPO/AirME cho aircraft maintenance (ICAPS 2022); v6 chỉ là bằng chứng component trong simulator hiện tại, không phải phát minh mới. V4/v5 capacity proposals vẫn là negative results. Setting synthetic H48+tail16, exogenous requests, N4/K2; chưa có production scheduling, service-time uncertainty, benchmark ngoài simulator hoặc transfer N/K.

Câu hỏi method tiếp theo, nếu triển khai, là liệu **tách RL chọn tập máy cần bảo trì khỏi một allocator technician có xét duration và opportunity cost** có giải quyết weak-seed reciprocal assignment. Trước vòng mới cần protocol fresh seeds, paired RL-from-scratch vs same actor/allocator ablation, cùng physics/objective/DEV rule, và kiểm tra policy có đổi action đủ thường xuyên. Không dùng TEST v6 để xác nhận vòng tiếp theo, không tăng PPO budget như trục nghiên cứu. Goal vẫn ACTIVE cho đến khi có bằng chứng method đáp ứng chất lượng điều phối đủ mạnh và giới hạn được chứng minh rõ hơn.

## Tái lập

```bash
uv run pytest -q tests/test_maintenance_framework_v6_policy.py tests/test_maintenance_framework_v1.py
uv run python -m ht_pdm_fjsp.maintenance_framework_v6_campaign --profile full --device cpu --output-dir "research_result/framework_v6_full_$(date -u +%Y%m%dT%H%M%SZ)"
uv run python research_result/verify_framework_v6.py --run-dir research_result/framework_v6_full_20261019T193747Z --output-dir "research_result/framework_v6_audit_$(date -u +%Y%m%dT%H%M%SZ)"
uv run python research_result/diagnose_framework_v6_permutation.py --run-dir research_result/framework_v6_full_20261019T193747Z --output-dir "research_result/framework_v6_permutation_$(date -u +%Y%m%dT%H%M%SZ)"
uv run python research_result/diagnose_framework_v6_wait.py --run-dir research_result/framework_v6_full_20261019T193747Z --output-dir "research_result/framework_v6_wait_$(date -u +%Y%m%dT%H%M%SZ)"
uv run python research_result/diagnose_framework_v6_matching.py --run-dir research_result/framework_v6_full_20261019T193747Z --output-dir "research_result/framework_v6_matching_$(date -u +%Y%m%dT%H%M%SZ)"
```

Re-running on seed 67200000–63 is technical replication, not new confirmation. For a new claim, register a new physical TEST split before examining it.
