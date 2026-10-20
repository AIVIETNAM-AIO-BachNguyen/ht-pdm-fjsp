# V7: RL chọn tập máy, shortest-duration technician allocator

Phân tích ngày 2026-10-20 (Asia/Taipei). Method proposal `research_result/method_v7_proposal_vi.md` và protocol `research_result/framework_v7_protocol_vi.md` được viết trước code/DEV/TEST; tests, Ubuntu smoke và independent smoke audit PASS. Input code/protocol/seed registry đã commit/push ở `8ede08148548df974a9d4fbddbbcaf8636a83764` **trước** FULL. Raw `research_result/framework_v7_full_20261020T012531Z/` và independent audit `research_result/framework_v7_audit_20261020T020506Z/independent_audit.json` **PASS**. Không đổi source, physics, cost, seed, budget hoặc checkpoint-selection rule sau khi mở TEST. Raw/checkpoint lớn giữ ngoài Git.

## Vấn đề, method và đối chứng

V6 cho thấy shared edge scorer tốt hơn matched-budget unshared edge scorer theo primary nhưng kém flat/risk/rollout pooled. Post hoc, vài weak shared replicas nhiều lần dùng assignment có cùng tập máy nhưng tổng duration dài hơn alternative legal, khiến shared technicians bận lâu. V7 dùng cấu trúc quyết định hai phần: MaskablePPO từ đầu vẫn **chọn tập máy cần phục vụ ngay** theo temporal cost; với tập máy đã chọn, exact allocator chỉ giữ legal matching có **tổng service duration hiện biết nhỏ nhất**; ties giữ nguyên. WAIT vẫn là một action. Đây là known-current-information mask, không nhìn realized future. Phần chọn request/máy qua thời gian là RL; bước exact assignment là thành phần hỗ trợ. Full H48+tail16 economic objective, request threshold, dynamics và tail risk controller không đổi.

Bốn scratch RL arms: `shared_free`, `shared_canonical` cùng 9.470 parameters; `flat_free`, `flat_canonical` cùng 17.046 parameters. Mỗi cặp cùng initial weights, cùng 100.352 PPO steps/train seed, cùng case/physical training tape, hai checkpoints 50.176/100.352. Năm training seeds ×4 arms =2.007.040 PPO steps. DEV 16 seeds/case chọn checkpoint **trong từng arm và từng training seed**; sau đó hash selection, mở TEST mới 64 physical seeds/case. Có 2.560 DEV và 5.888 TEST episodes (5.120 learned, 768 FCFS/risk/rollout), cùng four-case N4/K2 panel. Primary đã khóa: cost `shared_canonical − shared_free` cân bốn case trên 5×64 train-seed/physical-seed cells, hierarchical paired bootstrap 20.000 draws theo cả hai tầng; confirm nếu mean<0 và upper CI95<0. Baseline/case contrasts secondary descriptive.

## Kết quả đã kiểm toán

**Primary CONFIRMED:** `shared_canonical − shared_free` = **−47,695 cost**, hierarchical CI95 **[−85,442; −13,731]**. Trong 320 cells: 173 wins, 82 ties, 65 losses. Per-training-seed means 68000000–04: −0,156; −0,016; −80,043; −77,383; −80,875. Đây là conditional benefit: hai seeds đã tự chọn duration ngắn hầu như không đổi; ba seeds yếu cải thiện rõ. Không mô tả như gain đều ở mọi replica.

| Case | Flat free | Flat canonical | Shared free | Shared canonical | Risk | Rollout | Shared canonical − free (CI95 descriptive) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reciprocal scale2 | 829,53 | 828,81 | 1063,90 | 945,45 | 830,36 | 721,06 | −118,45 [−211,52; −35,51] |
| Interchangeable scale1 | 784,22 | 784,92 | 743,12 | 744,31 | 749,06 | 683,58 | +1,19 [−0,91; +4,88] |
| Reciprocal scale3 | 1038,11 | 1036,89 | 1225,58 | 1157,50 | 1041,48 | 1059,48 | −68,08 [−130,34; −16,34] |
| Specialist/generalist scale2 | 1014,46 | 1016,73 | 1003,27 | 997,82 | 1005,20 | 979,97 | −5,44 [−14,38; +2,38] |
| **Case-balanced** | **916,58** | **916,84** | **1008,97** | **961,27** | **906,53** | **861,02** | **−47,69 [−85,44; −13,73]** |

FCFS case-balanced 947,95. `flat_canonical − flat_free` = +0,261, CI95 [−8,167;+9,001]; no demonstrated generic gain from masking all architectures. `shared_canonical − flat_canonical` = +44,43 [−1,68;+89,62]. Shared canonical still **worse** than risk pooled by +54,74 [12,70;98,76] and rollout by +100,25 [55,38;145,66], secondary descriptive; no claim method beats good baselines. Rollout mean selection CPU time 0,675 s/episode vs shared canonical 0,028 s, risk 0,0014 s; compute differs. Aggregate H availability shared free 0,474 vs canonical 0,501; failures 8,18 vs 7,32 per episode. Flat free/canonical availability 0,537/0,539. Training wall totals: flat free 424 s, flat canonical 450 s, shared free 488 s, shared canonical 521 s on one CPU Torch thread.

## Cơ chế và giới hạn xác nhận

Independent verifier PASS source/artifact/checkpoint hashes, fresh splits/grid, independent paired training tape regeneration, within-seed DEV selection, initial-weight equality for each architecture pair, reward/cost reconciliation, request/service conservation, zero invalid assignments và all hierarchical bootstrap intervals. Nó replay **tất cả 5.120 learned TEST episodes/245.760 H decisions**, tái tạo đúng từng cost. Với mỗi state, verifier độc lập tính min duration trong mỗi legal machine subset, kiểm tra canonical action hợp lệ và đo free action lẽ ra bị loại.

Trên mỗi 12.288 H decisions/seed của `shared_free`, seeds 00/01 chọn **0** noncanonical assignments và cost hầu như không đổi ở canonical arm. Seeds 02/03/04 chọn **478/510/558** noncanonical assignments, với tổng excess duration 3.690/3.758/4.066 units. Canonical arms chọn 0 noncanonical. H service starts trên 256 TEST episodes/seed tăng từ 6.240→7.207, 6.193→7.225, 6.447→7.463 cho ba weak seeds; seeds 00/01 gần như không đổi. Reciprocal2 H starts/episode ở weak seed02 tăng 16,38→26,30, reciprocal3 11,89→16,91; vẫn dưới strong seeds khoảng 35 và 27. Điều này phù hợp với cơ chế ràng buộc duration làm giảm chiếm dụng technician và mở thêm cơ hội phục vụ. **Không** phải causal decomposition của từng quyết định: hai arms được train riêng từ đầu, mask thay đổi cả exploration, state trajectories và PPO gradients. Một same-weight inference intervention hoặc mediation study mới có thể tách các đường tác động.

Interchangeable case có durations đối xứng nên allocator ít tạo khác biệt; reciprocal2/3 cho gain lớn nhất vì worker speed khác biệt. Specialist case có thể có opportunity tradeoff: worker assignment ngắn nhất hôm nay không nhất thiết tốt nhất cho request tương lai. Cấu trúc min-total-duration không là dynamic dominance proof. Canonical shared vẫn thường under-service ở reciprocal cases so flat/risk, nên nguyên nhân còn lại có thể là chọn tập máy, temporal credit hoặc capacity opportunity; v7 chưa phân biệt được. Goal setting vẫn synthetic, exogenous PM threshold, N4/K2, no production jobs, uncertain service times, external simulator validation hay N/K transfer.

## Liên hệ literature và kết luận research

*Learning to Dispatch* (NeurIPS 2020) học RL dispatch priority trên graph và chứng minh transfer kích thước trong JSSP: [paper](https://papers.neurips.cc/paper_files/paper/2020/hash/11958dfee29b6709f48a9ba0387a2431-Abstract.html). HetGPO/AirME (ICAPS 2022) đã dùng heterogeneous graph RL cho predictive maintenance với shared crews: [paper](https://ojs.aaai.org/index.php/ICAPS/article/view/19839). V7 không phát minh graph dispatch, learned+matching hybrid hay shortest-duration assignment; đóng góp ở đây là **kiểm chứng một decomposition cụ thể** cho simulator request-lifecycle, với same-architecture free/canonical ablation, fresh paired panel và cơ chế/giới hạn được đo. Chưa đối chiếu toàn diện benchmarks/papers để claim novelty, scalability, optimality hoặc deployment readiness.

Câu hỏi đã khóa được trả lời: exact current-duration allocator có giá trị **khi** shared RL actor tự học assignment chậm; nó tránh một learning failure mode nhưng không sửa được toàn bộ dispatch và không thắng strong baselines pooled. V4/v5 opportunity features là negative results; v6 shared-head bias và v7 decomposition là conditional positive components. Framework RL/simulator/evaluation đủ chạy và tái lập, song hướng nghiên cứu tiếp theo nên kiểm tra *temporal choice of machine set / future technician opportunity* với proposal/seed mới, không dùng tuning PPO hoặc mở lại TEST v7 để cứu performance.

## Tái lập

```bash
uv run pytest -q tests/test_maintenance_framework_v*.py tests/test_maintenance_requests.py tests/test_maintenance_dispatch.py tests/test_maintenance_dispatch_policy.py tests/test_maintenance_request_experiment.py
uv run python -m ht_pdm_fjsp.maintenance_framework_v7_campaign --profile full --device cpu --output-dir "research_result/framework_v7_full_$(date -u +%Y%m%dT%H%M%SZ)"
uv run python research_result/verify_framework_v7.py --run-dir research_result/framework_v7_full_20261020T012531Z --output-dir "research_result/framework_v7_audit_$(date -u +%Y%m%dT%H%M%SZ)"
```

Reusing TEST 68200000–63 is technical replication only, not new confirmation. Exact full invocation, code hash, registry, selected checkpoint hashes, configurations, tapes and per-episode results are recorded in the raw run manifest.
