## Material Passport

- Origin: ARS experiment-agent; local fixed-budget experiment and independent statistics.
- Verification: VERIFIED 56 file bindings, committed source, frozen choices, episode grids, training provenance and four count-weight bootstrap intervals. Physical replay is runner evidence, not a second independently implemented simulator.

# Vòng 3: masked PPO học joint maintenance

Run `maintenance_joint_learning_mac_v1_full_cpu_20261012T064635Z`, commit `c300d10`, macOS ARM CPU. COMPLETED / audit PASS, 132.724 seconds. Protocol và tất cả hyperparameter được khóa trước kết quả full; smoke không dùng để chọn hiệu năng.

3 training replicas × 65.536 bước =196.608 bước học. Mỗi replica có 1.360 episode hoàn tất và 8 trajectory cuối dở dang, mỗi trajectory32 bước; tổng245.568 bước vật lý training. Giữ toàn bộ dữ liệu dở dang. Development32 seeds × 4 checkpoints × 3 replicas =384 episode; chọn checkpoint bằng mean cost development, tie chọn checkpoint sớm hơn. Lựa chọn lần lượt 49.152,65.536,65.536 bước được hash trước khi mở test. Test128 seed mới × 10 controllers =1.280 episode,76.800 bước vật lý. Test chỉ chạy một lần, không tăng seed theo kết quả.

| Controller | Cost H+L | Availability H | Pending H | Failures H+L |
|---|---:|---:|---:|---:|
| ppo seed 41000000 | 469.922 | 73.54% | 0.0234 | 1.570 |
| ppo seed 41000001 | 468.961 | 73.48% | 0.0156 | 1.469 |
| ppo seed 41000002 | 466.820 | 73.47% | 0.0156 | 1.336 |
| initial seed 41000000 | 789.977 | 63.02% | 0.2656 | 6.938 |
| initial seed 41000001 | 1021.008 | 57.61% | 0.7500 | 9.125 |
| initial seed 41000002 | 1310.141 | 46.90% | 0.6641 | 11.109 |
| deadline_matching | 473.086 | 71.35% | 0.0000 | 0.250 |
| isolated_h6 | 492.477 | 71.55% | 0.0234 | 1.523 |
| isolated_endpoint | 473.836 | 74.15% | 0.0000 | 2.273 |
| joint_rollout | 442.961 | 73.64% | 0.0000 | 0.453 |

| Contrast exploratory | Estimate | CI 95% |
|---|---:|---:|
| gain_vs_initial | 54.9617% | [41.4318; 64.1842]% |
| gain_vs_deadline | 0.9551% | [0.1640; 1.7203]% |
| availability_difference | 2.1512 pp | [1.5422; 2.8035] pp |
| pending_difference | 0.0182 | [0.0000; 0.0417] |

PPO trung bình cost 468.568 so với deadline473.086: giảm 0.9551%, CI 95%[0.1640%;1.7203%]. Availability tăng 2.1512 điểm phần trăm. Pending tại H tăng 0.0182 job/mỗi episode; tất cả backlog hết sau common12-tick tail. Giảm54.96% so với network ban đầu chỉ là kiểm tra học; baseline ban đầu rất yếu. Không dùng con số này làm đóng góp chính.

PPO còn đắt hơn joint rollout 25.607 đơn vị cost (~5.47% cost PPO). Rollout trên chính panel test này giảm 6.37% so với deadline. Không trộn baseline mean giữa panel 256 seed vòng2 và 128 seed vòng3. Ba PPO replica đều có mean cost thấp hơn deadline trên test, nhưng chỉ có 3 training seeds và bốn CI 95% chưa hiệu chỉnh: kết quả learning vẫn EXPLORATORY, không thay verdict xác nhận setting vòng2. Centralized PPO chọn một joint matching; đây chưa phải thuật toán MARL hoặc bằng chứng cần MARL.

Feasibility, model save/load, RNG preservation, mọi test action/state/return và metadata checkpoint được runner replay kiểm tra. Forecast scoring được tái tạo cho2 test seeds đầu theo protocol; không tuyên bố forecast replay cho mọi seed. Hash56 file, so khớp snapshot với commit, grid training/dev/test và CI được kiểm tra bằng script độc lập không gọi hàm summarize của runner.

11 fallacy checks: báo cáo từng replica và cùng panel (Simpson/ecological); disclosed fixed profile đã chọn trước fresh confirmation (Berkson/regression to mean); giữ WAIT, masked roots và mọi episode (collider/base rate/survivorship); budget/checkpoints/seeds khóa trước, bốn exploratory CI chưa hiệu chỉnh được ghi rõ (look elsewhere/forking paths); học policy trên simulator không suy ra causal effect của assignment hoặc necessity of MARL (causality/reverse causality). 11/11 checked.
