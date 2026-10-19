## Material Passport

- Origin: ARS experiment-agent; prospective Mac CPU experiment with native shared-context scientific review and independent numerical implementation.
- Date: 2026-10-12, Asia/Taipei.
- Verification: VERIFIED. 64 file bindings, 49 Git blobs, all 177,600 physical ticks, baseline/winner choices, fixed-subset root references and all 18 indexed-bootstrap intervals independently reconstructed. Native workers share context/model; this is not blinded or cross-model validation.

# Chốt setting làm rõ technician skill

**`cross_d3_q050_skewed` đạt CROSS_SKILL_SETTING_CONFIRMED.** Đây là setting phù hợp hơn với đóng góp phân công technician theo skill: có chuyên môn chéo, đánh đổi tốc độ–phục hồi và tải công việc lệch giữa hai người. Một winner được chọn trên screen và xác nhận trên 256 seed mới; cả 18 tiêu chí khóa trước đều PASS.

## Setting dễ hiểu

Có 4 máy và 2 technician. Trên máy 0–2, A sửa trong 1 tick nhưng chỉ phục hồi 50%; B cần 3 ticks nhưng phục hồi hoàn toàn. Trên máy 3, vai trò đảo lại: B nhanh/phục hồi 50%, A chậm/phục hồi hoàn toàn. Mọi cạnh đều hợp lệ.

| Máy | Technician A: duration, restoration | Technician B: duration, restoration |
|---|---|---|
| 0 | 1 tick, 0.50 | 3 ticks, 1.00 |
| 1 | 1 tick, 0.50 | 3 ticks, 1.00 |
| 2 | 1 tick, 0.50 | 3 ticks, 1.00 |
| 3 | 3 ticks, 1.00 | 1 tick, 0.50 |

Restoration là **giảm tuổi ảo lúc hoàn tất**, `floor(age × (1-restoration))`; không phải xác suất sửa thành công. Ví dụ tuổi 4: sửa nhanh còn tuổi 2, sửa chậm về tuổi 0. Tuổi không tăng khi đang service. Máy xuống cấp sau idle ticks; bắt đầu từ tuổi 4 có xác suất hỏng điều kiện 0.45/tick, max_age 8. PM chỉ khi tuổi≥3, CM khi failed. Mỗi tick chọn WAIT hoặc một partial matching khả thi giữa máy rảnh và technician rảnh.

48 ticks quyết định joint, sau đó 12 ticks dùng cùng deadline policy để giảm ảnh hưởng cuối horizon. Chi phí PM 1, CM 2, downtime 6/machine-tick, failure 15; failed waiting vượt 4 ticks bị tính thêm 12/tick. Các chi phí và hazard giữ như setting cũ; chỉ thay service/restoration matrix.

Ba máy đầu có cùng hướng nhanh/chậm nên tranh chấp technician A khi ưu tiên sửa nhanh, hoặc B khi ưu tiên phục hồi hoàn toàn. Policy phải cân bằng downtime ngay bây giờ, tuổi còn lại, availability của từng technician cho công việc sau và trạng thái máy. “Luôn giao người nhanh nhất” hay “luôn giao người phục hồi tốt nhất” không thể hiện đủ bài toán này.

## Kết quả xác nhận trên cùng 256 seed

| Controller | Cost H+L | Cost H | Availability H | Pending H | Pending H+L |
|---|---:|---:|---:|---:|---:|
| deadline_matching | 679.906 | 541.539 | 62.77% | 0.0586 | 0.0430 |
| isolated_h6 | 716.859 | 578.848 | 59.01% | 0.1719 | 0.0703 |
| isolated_endpoint | 678.441 | 540.570 | 63.13% | 0.0000 | 0.0000 |
| assignment_rollout | 619.543 | 500.688 | 64.51% | 0.0156 | 0.0117 |
| joint_rollout | 585.605 | 473.605 | 66.34% | 0.0000 | 0.0000 |

Assignment rollout giữ nguyên tập máy/start timing do baseline chọn **tại trạng thái hiện tại**, chỉ chọn lại technician bằng lookahead. Cost H+L giảm **8.8782%**, simultaneous CI 99.7222% **[7.1424%;10.6657%]** so với deadline baseline đã khóa bằng development. Trên test, isolated_endpoint có mean cost hơi thấp hơn deadline (678.441 so với 679.906); không rerank baseline sau test. Tất cả comparator được giữ trong bảng.

Joint rollout giảm **13.8697%** so với deadline, **18.3096%** so với isolated_h6 và **13.6837%** so với isolated_endpoint; ba CI lower lần lượt 12.2750%,16.5177%,12.0656%, đều vượt practical margin 1%. Assignment H-only gain 7.5436%, joint H-only gain 12.5445%; lợi ích không chỉ nằm ở tail. Assignment availability 64.51%, joint 66.34%, đều tăng so với deadline 62.77%; pending guards đều PASS. Một ít pending vẫn còn ở assignment tail, được giữ và báo cáo đầy đủ.

## Bằng chứng riêng cho phân công

Ở mỗi baseline landmark, giữ **cùng trạng thái, tick và tập máy**; chọn một allocation trên selection bank, rồi dùng holdout bank độc lập so với baseline, chất lượng(duration/restoration) và endpoint isolated-DP allocation trên đúng tập máy đó. Tất cả 1,792 clock landmarks đều được giữ, kể cả WAIT/no-choice/không đổi action. Bootstrap resample 256 physical seeds, giữ 7 landmarks của mỗi seed cùng nhau.

Mean heldout gain so với endpoint-DP allocation là **5.5805 cost unit/common landmark**, CI 99.7222% **[3.8213;7.5159]**, vượt margin 0.5 đã khóa trước. So với baseline/quality là 5.7072, CI[3.9315;7.6267]. Đây là marginal value của **assignment ban đầu** dưới một continuation rule cố định nhưng state-adaptive; các quyết định tương lai có thể đổi do assignment làm thay đổi occupancy/restoration. Không diễn giải 8.88% closed-loop gain thành cả 48 ticks có lịch máy/timing giống hệt nhau. Không nhân mean local gain với 7 để suy ra episode savings: horizons landmark chồng lấn.

Có 1047/1792 landmarks có từ 2 allocation hợp lệ (58.4263%, CI[53.0692%;63.7835%]); selected root đổi ở 201 landmarks. Spread tổng tuổi ảo sau completion khác 0 ở 929 landmarks, được tính bằng tuổi phục hồi thực tế sau floor; nominal chất lượng khác nhau không tự động được coi là khác outcome.

| Tiêu chí | Estimate | CI 99.7222% | Gate |
|---|---:|---:|---|
| local_baseline | 5.7072 cost | [3.9315; 7.6267] cost | PASS |
| local_quality | 5.7072 cost | [3.9315; 7.6267] cost | PASS |
| local_endpoint_dp | 5.5805 cost | [3.8213; 7.5159] cost | PASS |
| assignment_cost_gain | 8.8782% | [7.1424; 10.6657]% | PASS |
| joint_cost_vs_deadline_matching | 13.8697% | [12.2750; 15.5077]% | PASS |
| joint_cost_vs_isolated_h6 | 18.3096% | [16.5177; 20.0266]% | PASS |
| joint_cost_vs_isolated_endpoint | 13.6837% | [12.0656; 15.2845]% | PASS |
| assignment_availability_difference | 1.7395 pp | [1.1938; 2.3406] pp | PASS |
| assignment_pending_difference | -0.0430 job | [-0.0938; -0.0039] job | PASS |
| assignment_absolute_availability | 64.5060% | [63.9486; 65.0167]% | PASS |
| assignment_absolute_pending | 0.0156 job | [0.0000; 0.0430] job | PASS |
| assignment_interior_gain | 7.5436% | [5.8709; 9.2711]% | PASS |
| joint_availability_difference | 3.5726 pp | [2.9089; 4.2664] pp | PASS |
| joint_pending_difference | -0.0586 job | [-0.1094; -0.0195] job | PASS |
| joint_absolute_availability | 66.3391% | [66.2048; 66.4754]% | PASS |
| joint_absolute_pending | 0.0000 job | [0.0000; 0.0000] job | PASS |
| joint_interior_gain | 12.5445% | [10.9666; 14.1828]% | PASS |
| assignment_choice_fraction | 58.4263% | [53.0692; 63.7835]% | PASS |

## Screening, controls và các cấu hình không đạt

8 selectable profiles: d∈{2,3}, fast restoration∈{0.25,0.50}, balanced2+2/skewed3+1, slow restoration1.00. Baseline chọn bằng 16 development seeds/cấu hình; screen dùng 24 seed khác; chỉ winner được mở confirmation 256 seed. Candidate xếp hạng theo min(assignment gain, joint gain vs mỗi cheap rule), phải vượt tất cả screen guards trước. Bảy selectable profiles không đạt được giữ nguyên; một số có saving tốt nhưng availability/backlog guard thất bại. Không thay margin để nhận chúng.

| Profile | Assignment gain | Joint gain vs deadline | Local gain vs endpoint-DP | Không đạt tiêu chí screen |
|---|---:|---:|---:|---|
| cross_d2_q025_balanced | 0.079% | 0.257% | 0.040 | assignment_cost_gain, joint_cost_vs_deadline_matching, local_baseline, local_quality, local_endpoint_dp, joint_interior_gain |
| cross_d2_q025_skewed | 2.763% | 3.509% | 1.516 | joint_pending_difference |
| cross_d2_q050_balanced | 3.600% | 12.871% | 3.523 | assignment_pending_difference |
| cross_d2_q050_skewed | 1.161% | 1.271% | 0.421 | assignment_cost_gain, joint_cost_vs_deadline_matching, local_baseline, local_quality, local_endpoint_dp |
| cross_d3_q025_balanced | 0.000% | 0.520% | 0.000 | assignment_cost_gain, joint_cost_vs_deadline_matching, joint_cost_vs_isolated_h6, local_baseline, local_quality, local_endpoint_dp, assignment_interior_gain |
| cross_d3_q025_skewed | 1.292% | 4.997% | 0.385 | assignment_cost_gain, local_baseline, local_quality, local_endpoint_dp |
| cross_d3_q050_balanced | 7.489% | 12.078% | 7.383 | assignment_availability_difference |
| cross_d3_q050_skewed | 12.218% | 16.789% | 7.080 | PASS |
| old_fast1_slow325 | -0.034% | 9.434% | 0.006 | assignment_cost_gain, local_baseline, local_quality, local_endpoint_dp, assignment_interior_gain |
| exchangeable | 0.000% | 2.590% | 0.000 | assignment_cost_gain, local_baseline, local_quality, local_endpoint_dp, assignment_interior_gain |

Screen controls là descriptive, không phải fresh-confirmation comparison giữa profile: setting cũ assignment gain −0.034%, localDP gain 0.006; exchangeable control assignment và mọi local gain đúng 0. Các số này hỗ trợ kiểm tra cơ chế nhưng không phải estimate chính. Old fast1_slow325 vẫn có verdict joint phù hợp của vòng trước; không xóa hay phủ nhận kết quả đó.

## Kiểm chứng và giới hạn

Code/source `8672acb` được commit/push trước full run. Complete suite 904 passed trên initial implementation, final 22 targeted passed sau audit revisions và smoke cuối PASS. Full Mac run 508.687s, tổng 480 dev + 1200 screen + 1280 confirmation episodes, 177,600 physical ticks và 3,472 landmarks. Runner tái tạo planner forecast/banks cho 2 raw seeds đầu mỗi phase/config; mọi physical tick và root grid đều kiểm tra. Verifier riêng không import simulator/runner: tự dựng initial shocks, scalar transition, restricted DP, cheap/root references, baseline/winner và 18 bootstrap intervals bằng indexed seed blocks thay count-weight implementation. Tất cả 64 hash/size và 49 committed blobs khớp.

Đây là fixed synthetic maintenance benchmark, generalization hiện chỉ sang shock seeds mới. Chưa chứng minh generalization sang matrices khác, số máy lớn hơn, full production/FJSP system hoặc real plant. Known-model rollout chưa phải global optimality. **Search này chưa train policy**, nên chưa tuyên bố RL học được phân công, MARL cần thiết hoặc learner thắng baseline. Setting giờ có assignment headroom đủ rõ để đánh giá thuật toán học bước sau. Báo cáo 18 simultaneous CI là approximate per-protocol; không bảo đảm overall error cho mọi adaptive research round trước/sau.

11 fallacy checks: method-specific means cùng paired panel (Simpson/ecological); finite profile selection công khai và fresh confirmation, screen winner giảm 12.22%→8.88% cũng giữ nguyên (Berkson/regression to mean); tất cả clock/no-choice/root coverage và losses retained (collider/base rate/survivorship); locked panels, margins,18-family, controls không được chọn (look elsewhere/forking paths); root intervention tách khỏi mediated closed-loop effects/occupancy, không suy ra causal superiority của một feature hay MARL (causality/reverse causality). 11/11 checked.

Không cần search tiếp cho mục tiêu hiện tại: setting đã vượt đủ tiêu chí phân công riêng và joint operational guards trên seed mới. Không tune thêm trên confirmation panel đã mở. Raw artifacts và setting cũ được giữ nguyên; thực thi full trên Mac theo yêu cầu trực tiếp, không SSH/lab transfer.

## Local artifacts

- [Setting JSON](</Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_cross_skill_search_20261012/selected_setting.json>)
- [Independent verification](</Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_cross_skill_search_20261012/round1/analysis.json>)
- [Independent verifier source](</Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_cross_skill_search_20261012/round1/verify.py>)
- [Raw manifest](</Users/bachng/Coding/Reinforcement Learning/marl/artifacts/maintenance_cross_skill_mac_v1_full_cpu_20261012T080148Z/manifest.json>)
- [Locked protocol snapshot](</Users/bachng/Coding/Reinforcement Learning/marl/artifacts/maintenance_cross_skill_mac_v1_full_cpu_20261012T080148Z/protocol.md>)
- [Previous setting](</Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_mac_search_20261012/selected_setting.md>)
