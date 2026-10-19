# Cấu hình 2–2 đã đạt đủ 18 guard

## Material Passport

Dữ liệu tổng hợp từ maintenance simulator, chạy CPU trên Mac. Đây là tìm kiếm
một setting với development, screening và confirmation seed tách biệt; không
có dữ liệu bên ngoài, người tham gia hoặc training RL. Scientific review và
verifier được thực hiện bằng native worker trong cùng ngữ cảnh. Verifier dùng
scalar physics và bootstrap viết riêng khỏi runner, được kế thừa và chỉnh theo
protocol; không phải đánh giá mù hoặc cross-model.

Verification Status: VERIFIED cho artifact, source, physical replay và phép tính
thống kê. Chưa có field validation hoặc bằng chứng về thuật toán học.

**Kết luận: CROSS_SKILL_SETTING_CONFIRMED, PASS 18/18; independent audit PASS.**
Đã đạt điều kiện dừng do người nghiên cứu yêu cầu. Không cần mở vòng tìm kiếm
khác để hoàn thành yêu cầu này. Các lần fail trước vẫn được giữ nguyên.

## Setting chốt: balanced_a2_b4_u12

Bốn máy, hai technician. A nhanh trên M1–M2; B nhanh trên M3–M4. Cả hai đều làm
được mọi máy. Hai technician có cùng phân bố độ khó, đối xứng qua hai nhóm máy.

| Máy | Technician A: thời gian / phục hồi | Technician B: thời gian / phục hồi |
|---|---|---|
| M1 | 1 tick / 50% | 2 tick / 100% |
| M2 | 1 tick / 50% | 4 tick / 100% |
| M3 | 2 tick / 100% | 1 tick / 50% |
| M4 | 4 tick / 100% | 1 tick / 50% |

Làm nhanh giải phóng technician sớm nhưng phục hồi ít hơn. Tại tuổi bảo trì 3,
phương án nhanh đưa tuổi về floor(3×.5)=1; phương án chậm đưa về 0. Tác động thực
còn phụ thuộc tuổi máy vì simulator làm tròn xuống.

Giữ H=48 và 12 tick tail chung, không reset. Failure age=4, probability=.45,
max age=8; PM cost=1, CM=2, downtime=12 mỗi machine-tick, failure=15, waiting
limit=4 và overdue price=12. Chỉ máy idle đang failed hoặc age>=3 mới được bắt
đầu; technician phải rảnh. Có WAIT và mọi partial matching hợp lệ.

**Thay đổi cần ghi rõ:** downtime cost tăng từ 6 lên 12 trong vòng hai. Đây là
một lựa chọn objective trong môi trường tổng hợp, chưa được hiệu chỉnh theo giá
của nhà máy thật. Các cost khác và health guard không đổi.

## Xác nhận trên 256 seed mới

Baseline **isolated_h6** được khóa từ 24 development seed, theo mean H+L cost
thấp nhất. Screening dùng 32 seed riêng. Confirmation dùng raw seed
45200000–45200255; forecast 128 CRN scenario/root và bootstrap 160.000 paired
seed block. Confidence mỗi CI **99,930556%** = 1−.0125/18, chặt hơn lượt trước.

- **Allocation-only:** giảm H+L cost **3,5708%**, CI [2,6625%; 4,5260%]. H-only
  gain cũng dương: 2,3975%, CI [1,4363%; 3,4175%].
- **Joint:** giảm **4,6699%** so với baseline h6, CI [3,5606%; 5,8390%]; giảm
  **9,6830%** so với deadline, CI [8,0216%; 11,3070%]; giảm **4,9307%** so với
  endpoint, CI [3,8544%; 6,0732%].
- Availability allocation **66,4185%**, baseline **66,9820%**: giảm **0,5636 điểm
  phần trăm**, CI [−0,7711; −0,3816]. Joint **66,7684%**: giảm **0,2136 điểm
  phần trăm**, CI [−0,3479; −0,0895]. Đây là giảm availability có bằng chứng,
  không phải tăng uptime. Cả hai cận dưới vẫn vượt margin **−1 điểm phần trăm**
  đã khóa, nên guard noninferiority đạt.
- Pending tại H bằng 0 cho cả hai planner trên toàn bộ confirmation episode.
  Mean pending sau tail: allocation .01171875, joint 0.
- Local assay giữ nguyên state, tick và subset ban đầu: gain **7,9182 cost mỗi
  common landmark**, CI [5,7057; 10,3559]. Ba reference baseline/quality/endpoint
  chọn **cùng một allocation trên toàn bộ confirmation state**. Vì vậy ba gate
  này là so sánh trùng nhau, không phải ba bằng chứng độc lập. Gain là giá trị
  dự kiến đến hết horizon của intervention hiện tại, không phải tiết kiệm thực
  mỗi tick hoặc tác động của một lịch cả episode bị cố định.
- Có lựa chọn assignment tại **1255/1792 landmark = 70,0335%**; rollout đổi root
  tại 298 landmark. Duration span dương tại 1255, restoration span dương tại
  1236, tổng tuổi sau phục hồi có span dương tại 1237. WAIT và no-choice zero
  được giữ đầy đủ; bảy landmark được lấy mean trong từng raw seed.

Cả 18 guard đều bắt buộc; không nới ngưỡng hoặc chỉ chọn một phần kết quả đạt.
Chưa thể suy ra PPO/MARL sẽ đạt gain tương tự, rollout là tối ưu, hoặc setting
đã được xác nhận ngoài simulator. Vòng này không tách riêng tác động của duration
heterogeneity với downtime pricing; các so sánh cơ chế giữa candidate là exploratory.

## Controller means

| Controller | H+L cost | Availability tại H | Pending tại H | Tail pending |
|---|---:|---:|---:|---:|
| deadline_matching | 1157.309 | 66.6809% | 0.0820 | 0.0898 |
| isolated_h6 | 1096.449 | 66.9820% | 0.0000 | 0.0391 |
| isolated_endpoint | 1099.457 | 66.9535% | 0.0000 | 0.0234 |
| assignment_rollout | 1057.297 | 66.4185% | 0.0000 | 0.0117 |
| joint_rollout | 1045.246 | 66.7684% | 0.0000 | 0.0000 |

## Physical outcomes và cost components

H có 192 machine-tick; H+L có 240. Số tick ngừng máy và số failure được báo riêng
với tiền cost. Không so raw local gain giữa giá downtime 6/9/12 như thể đó là cải
thiện vật lý. Các component mean dưới đây là descriptive, không thêm claim về CI.

| Controller | Unavailable tick H | Unavailable tick H+L | Failure H+L | Overdue tick H+L | Maintenance cost | Failure cost |
|---|---:|---:|---:|---:|---:|---:|
| assignment_rollout | 64.4766 | 80.4844 | 0.9141 | 0.0000 | 77.773 | 13.711 |
| deadline_matching | 63.9727 | 79.9883 | 7.5312 | 0.0000 | 84.480 | 112.969 |
| isolated_endpoint | 63.4492 | 79.0938 | 4.8750 | 0.0000 | 77.207 | 73.125 |
| isolated_h6 | 63.3945 | 79.4102 | 4.4453 | 0.0000 | 76.848 | 66.680 |
| joint_rollout | 63.8047 | 79.8047 | 0.6367 | 0.0000 | 78.039 | 9.551 |

Tradeoff thể hiện rõ: so với h6, allocation có nhiều unavailable tick hơn một
chút nhưng failure H+L giảm từ 4,4453 xuống 0,9141; joint còn 0,6367. Các số này
là mean trên simulator; guard availability vẫn phải đạt độc lập.

## Evidence và tái lập

Source commit `2b1cf5a8bb55e8d69ed192fc8ee4e34ae01bd9ec`, branch
`codex/maintenance-balanced-search-mac-v2`, đã push trước full run.
Tests **42 PASS trong 39,49s**, gồm scalar/vector forecast với cả hai giá downtime
và duration khác nhau. CLI smoke **10,342s PASS**; smoke mới trên committed source
cũng qua independent verifier. Không chạy thêm test lặp lại sau khi code đã ổn.

Full CPU Mac hoàn tất **647.296s**: 720 development + 1600
screening + 1280 confirmation = **3600 episode**, **216.000 physical tick**,
4032 landmark, 6970 root row, 812.967.808 forecast transition; không có invalid
assignment. Runner replay toàn bộ physical state/event/cost và root grid,
tái tạo forecast bank trên hai raw seed đầu mỗi phase/config.

Independent verifier PASS: **69 artifact hash**, **54 committed source blob**,
đúng phase grid, frozen baseline, health-first winner, expanded historical seed
registry; scalar physical replay và **18 CI được tính lại bằng direct indexing**
các seed block. Native claim review cũng PASS. Không dùng verifier như một
chứng nhận tuyệt đối về mọi giả định của nghiên cứu.

Artifact: `/Users/bachng/Coding/Reinforcement Learning/marl/artifacts/maintenance_balanced_search_mac_v2_full_cpu_20261012T093417Z`

```bash
cd /private/tmp/marl-maintenance-assignment-confidence-v1 && .venv/bin/python scripts/verify_maintenance_balanced_search_v2.py '/Users/bachng/Coding/Reinforcement Learning/marl/artifacts/maintenance_balanced_search_mac_v2_full_cpu_20261012T093417Z' /private/tmp/marl-maintenance-assignment-confidence-v1 --output /private/tmp/balanced_v2_reverification_$(date -u +%Y%m%dT%H%M%SZ).json
```

## Đủ 18 comparison đã khóa

Cost gain và availability trong bảng là fraction; local gain là cost/common
landmark. Bootstrap theo whole paired seed, không coi landmark là mẫu độc lập.
Round một của series mới dùng family alpha .025 và fail screening; round hai
.0125, không recycle phần alpha chưa dùng. Bootstrap/spending là approximate,
không điều chỉnh ngược các lượt research trước hoặc bảo đảm exact coverage.

| Metric | Estimate | CI 99,930556% | Gate |
|---|---:|---|---|
| local_baseline | 7.918239 | [5.705696, 10.355870] | PASS |
| local_quality | 7.918239 | [5.705696, 10.355870] | PASS |
| local_endpoint_dp | 7.918239 | [5.705696, 10.355870] | PASS |
| assignment_cost_gain | 0.035708 | [0.026625, 0.045260] | PASS |
| joint_cost_vs_deadline_matching | 0.096830 | [0.080216, 0.113070] | PASS |
| joint_cost_vs_isolated_h6 | 0.046699 | [0.035606, 0.058390] | PASS |
| joint_cost_vs_isolated_endpoint | 0.049307 | [0.038544, 0.060732] | PASS |
| assignment_availability_difference | -0.005636 | [-0.007711, -0.003816] | PASS |
| assignment_pending_difference | 0.000000 | [0.000000, 0.000000] | PASS |
| assignment_absolute_availability | 0.664185 | [0.661836, 0.666341] | PASS |
| assignment_absolute_pending | 0.000000 | [0.000000, 0.000000] | PASS |
| assignment_interior_gain | 0.023975 | [0.014363, 0.034175] | PASS |
| joint_availability_difference | -0.002136 | [-0.003479, -0.000895] | PASS |
| joint_pending_difference | 0.000000 | [0.000000, 0.000000] | PASS |
| joint_absolute_availability | 0.667684 | [0.666239, 0.669047] | PASS |
| joint_absolute_pending | 0.000000 | [0.000000, 0.000000] | PASS |
| joint_interior_gain | 0.036667 | [0.025197, 0.048779] | PASS |
| assignment_choice_fraction | 0.700335 | [0.662946, 0.738839] | PASS |

## Screening giữ đầy đủ

Trong số ứng viên đủ strict screening gate, xếp theo min của hai planner về
paired availability mean−2 sample SE; tie theo min cost gain rồi candidate ID.
Đây chỉ là ranking heuristic, không phải một CI. Hai a2/b4 u9/u12 đủ điều kiện;
u12 là winner đã khóa trước khi mở confirmation. Không mở runner-up.

| Profile | Eligible | Health rank score | Assignment cost gain | Failed screen metrics |
|---|---|---:|---:|---|
| balanced_a2_b2_u09 | False | -0.002043 | 1.401% | assignment_cost_gain, joint_cost_vs_isolated_h6 |
| balanced_a2_b2_u12 | False | -0.001913 | 0.953% | assignment_cost_gain, joint_cost_vs_isolated_h6 |
| balanced_a2_b4_u09 | True | -0.005891 | 3.446% | none |
| balanced_a2_b4_u12 | True | -0.005691 | 2.479% | none |
| balanced_a3_b3_u09 | False | -0.016393 | 9.933% | assignment_availability_difference |
| balanced_a3_b3_u12 | False | -0.015509 | 7.779% | assignment_availability_difference |
| balanced_a3_b5_u09 | False | -0.021503 | 8.253% | assignment_availability_difference, joint_availability_difference |
| balanced_a3_b5_u12 | False | -0.023107 | 6.145% | assignment_availability_difference, joint_availability_difference |
| old_balanced_d3_q050 | False | -0.040027 | 7.583% | assignment_availability_difference |
| exchangeable | False | -0.000468 | 0.000% | assignment_cost_gain, joint_cost_vs_isolated_endpoint, local_baseline, local_quality, local_endpoint_dp, assignment_interior_gain |

## ARS statistical fallacy scan: 11/11 checked

1. **Simpson's paradox:** claim về pooled paired seed panel đã khóa; không suy ra
   mọi subgroup đều cải thiện. Chưa có kiểm tra subgroup thực địa.
2. **Ecological fallacy:** đơn vị inference là episode/seed của simulator; không
   suy ra lợi ích cá nhân technician hoặc từng máy từ pooled result.
3. **Berkson's paradox:** screening có chọn candidate, nhưng confirmation là panel
   mới; giữ đủ episode và landmark, không lọc chỉ state có gain.
4. **Collider bias:** local assay định nghĩa state distribution của baseline,
   không điều chỉnh regression theo outcome để suy ra effect toàn hệ thống.
5. **Base rate neglect:** failure probability .45 là conditional theo age; báo
   số failure thực và unavailable tick, không dùng .45 như prevalence toàn tick.
6. **Regression to mean:** không dùng khác biệt giữa các vòng/seed panel như
   effect cải thiện; mỗi comparison confirmation có paired baseline cùng seed.
7. **Survivorship bias:** giữ tất cả failed candidate, WAIT/no-choice state và
   đúng toàn bộ episode grid; không chỉ phân tích episode có lợi.
8. **Look-elsewhere effect:** all 18 gate và fresh sealed winner được giữ; alpha
   spending chỉ cho series mới. Không claim ba reference trùng là independent.
9. **Garden of forking paths:** versioned protocol khóa design/ranking/seed/budget
   trước run; giữ vòng một fail, không đổi guard sau khi xem confirmation.
10. **Correlation/causation:** local simulation intervention chỉ cố định current
    root; không gán riêng gain cho heterogeneity/pricing hoặc suy ra plant effect.
11. **Reverse causality:** action trước transition trong simulator; chưa suy ra
    hướng nhân quả của các association thực địa chưa được đo.
