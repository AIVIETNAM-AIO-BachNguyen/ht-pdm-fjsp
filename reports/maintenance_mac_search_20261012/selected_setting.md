## Material Passport

- Origin: ARS experiment-agent; three prospective Mac CPU experiments.
- Verification: VERIFIED engineering gates, runner physical replay, independent artifact/source/grid/CI verification. Confirmation applies to the fixed joint setting; PPO pilot remains exploratory.

# Setting được chốt cho joint maintenance

**Chọn `overlap3__fast1_slow325`: 4 machines, 2 technicians, joint decisions trong 48 ticks và 12-tick common tail.** Setting vượt cả 8 tiêu chí đã đăng ký trên 256 seed xác nhận mới. PPO sau đó học được policy có tín hiệu cải thiện trên một panel 128 seed khác. Đây là benchmark phù hợp để phát triển bài joint maintenance ở quy mô nhỏ.

| Thành phần | Setting khóa |
|---|---|
| Service duration theo machine × technician | `[[1,0],[1,3],[1,2],[1,5]]`; zero là cạnh không khả thi |
| Restoration | 0.75 trên mọi cạnh hợp lệ |
| Failure | từ age 4, xác suất điều kiện 0.45/tick; max_age 8 |
| Root action | WAIT hoặc partial matching khả thi; PM chỉ age≥3, CM khi failed |
| Chi phí | PM 1; CM 2; unavailable 6/machine-tick; failure 15 |
| Deadline | failed waiting limit 4; overdue 12/tick |
| Học | masked joint catalogue 21 actions, observation 71 features, current state và config |
| Primary metric | undiscounted cost H+L; tất cả policy dùng deadline matching trong tail |

Nội dung nghiên cứu nên tập trung **học ưu tiên chọn máy, thời điểm WAIT/service và phân bổ technician dưới capacity/skill constraints**. Kết quả hiện tại phù hợp joint decision; lợi ích assignment riêng rất nhỏ. Chưa đủ cơ sở đặt luận điểm chính là “học chọn technician thắng structured allocation”.

## Evidence để dừng search

Vòng 1 giữ nguyên verdict assignment-primary **NOT_CONFIRMED**; joint secondary vượt myopic nhưng chưa đủ để chốt setting. Vòng 2 dùng protocol joint mới và fresh seeds, chọn baseline bằng 32 development seeds rồi khóa trước 256 confirmation seeds. Không đổi profile/horizon/gates sau khi mở test.

| Joint rollout so với | Giảm cost H+L | CI 99.375% |
|---|---:|---:|
| deadline_matching | 5.3948% | [4.0623%;6.8000%] |
| isolated_h6 | 8.3928% | [6.7704%;10.0486%] |
| isolated_endpoint | 5.6693% | [4.0276%;7.3134%] |

Cả 3 cost CI có lower bound≥1%. Joint availability 73.56%, tăng 2.004 điểm phần trăm so với baseline đã chọn; pending tại H bằng 0 trên panel này. Interior H cost giảm 4.8793%, CI[3.6516%;6.1715%], nên lợi ích không chỉ nằm ở tail. Availability/pending difference và absolute guards đều PASS. Có 1.280 confirmation episodes và 1.792 landmarks; không bỏ episode lỗi hay landmark không thay action.

Subset rollout với structured duration matching có cost 445.254, rất gần joint rollout 444.441. Trên trajectory joint, subset/timing changes chiếm 1.4323% ticks; assignment-only changes 0.0488%. Đây là lý do giữ contribution ở joint maintenance, ưu tiên timing/selection; không suy diễn assignment là nguyên nhân chính.

## Pilot học sau khi khóa setting

3 MaskablePPO replicas × 65.536 bước, một bộ hyperparameter khóa trước: MLP[64,64],8 envs,rollout 256/env,batch 128,4 epochs,lr 3e-4,gamma 1,GAE 1,clip 0.2,entropy 0.01,value 0.5,maxgrad 0.5,reward −cost/20. Không mở test để tune. Checkpoint chọn theo 32 development seeds; test 128 seed mới.

| Controller, cùng pilot test panel | Cost H+L | Availability H | Pending H |
|---|---:|---:|---:|
| deadline_matching | 473.086 | 71.35% | 0 |
| PPO seed 41000000 | 469.922 | 73.54% | 0.0234 |
| PPO seed 41000001 | 468.961 | 73.48% | 0.0156 |
| PPO seed 41000002 | 466.820 | 73.47% | 0.0156 |
| PPO mean 3 replicas | 468.568 | 73.50% | 0.0182 |
| joint_rollout | 442.961 | 73.64% | 0 |

PPO giảm cost 0.9551% so với deadline, exploratory crossed-bootstrap CI 95%[0.1640%;1.7203%]; availability tăng 2.1512pp, CI[1.5422;2.8035]pp. Pending tăng 0.0182, CI[0;0.0417] jobs/episode; tail pending 0 cho mọi PPO episode. Cả 3 replica mean tốt hơn baseline, nhưng còn 25.607 cost gap tới rollout. Learning inference latency khoảng 0.00648s/episode trên 48 policy ticks, so với rollout 0.37816s; đây là measured controller time trên run này, không phải benchmark tốc độ đa máy.

Bốn 95% CI của pilot chưa hiệu chỉnh và chỉ có 3 training seeds: learning result là exploratory. Setting verdict lấy từ vòng2 với8 simultaneous 99.375% CI; không gộp hai panel hoặc biến pilot thành confirmatory claim. Network ban đầu yếu nên gain 54.96% so với initial chỉ chứng minh có học, không phải headline.

## Phạm vi và kết quả được lưu

Đây là một fixed synthetic maintenance profile với fresh stochastic shocks, chưa xác nhận tổng quát hóa sang service matrices khác, nhiều machines/technicians hoặc full production/FJSP system. Known-model rollout chưa chứng minh global optimality; centralized PPO chưa chứng minh MARL cần thiết. Skill/capacity constraints tạo coordination, nhưng assignment-only contribution chưa mạnh. Zero observed pending không có nghĩa xác suất backlog bằng 0. Bootstrap là approximate per-protocol; không bảo đảm sai số toàn bộ adaptive research search.

Tests: frontier full suite 867 passed; learning full suite 880 passed, sau audit revisions final targeted 12 passed và smoke mới PASS. Runner audit replay mọi physical evaluation state/action/return; planner forecast regeneration giới hạn 2 confirmation/test seeds đầu theo protocol. Phân tích độc lập kiểm tra hash/source/seed grids/selection/CI, không tuyên bố dùng một physical simulator độc lập thứ hai.

Code và protocol đã commit/push: `5f14d93` (boundary), `a769189` (frontier), `c300d10` (learning). Thực thi full trên Mac theo yêu cầu trực tiếp; không có SSH hoặc transfer từ lab. Các run cũ và thất bại được giữ nguyên. Stopping rule đã đạt: một setting joint có practical headroom được xác nhận và positive exploratory learning signal; không tiếp tục tune trên test panel đã mở.

11 fallacy checks: từng method/replica và cùng paired seeds (Simpson/ecological); disclosed post-screen fixed profile + fresh prospective panels (Berkson/regression to mean); all states/zero-gain landmarks và losses retained (collider/base rate/survivorship); fixed budgets và all eight gates, pilot uncorrected labeled (look elsewhere/forking paths); policy benefit không đồng nhất causal attribution/occupancy hoặc assignment (causality/reverse causality). 11/11 checked.

## Artifact links

- [Setting JSON](</Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_mac_search_20261012/selected_setting.json>)
- [Round1 report](</Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_mac_search_20261012/round1/report_vi.md>)
- [Round2 report](</Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_mac_search_20261012/round2/report_vi.md>)
- [Round3 report](</Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_mac_search_20261012/round3/report_vi.md>)
- [Confirmation manifest](</Users/bachng/Coding/Reinforcement Learning/marl/artifacts/maintenance_joint_frontier_mac_v1_full_cpu_20261012T062329Z/manifest.json>)
- [Learning manifest](</Users/bachng/Coding/Reinforcement Learning/marl/artifacts/maintenance_joint_learning_mac_v1_full_cpu_20261012T064635Z/manifest.json>)
- [Independent learning analysis](</Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_mac_search_20261012/round3/analyze.py>)
