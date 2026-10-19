# V3: rollout teacher, imitation và PPO fine-tuning

**Verdict đã đăng ký:** H1 đạt, H2 chưa đạt; điều kiện hai giả thuyết đồng
chính **không đạt**. Rollout-assisted PPO tốt hơn scratch PPO trên anchor,
nhưng chưa chứng minh PPO fine-tuning tốt hơn policy imitation-only. Không sửa
protocol, cost, case hoặc checkpoint sau khi mở TEST. Protocol trước code và
TEST: `framework_v3_protocol_vi.md`; source đã push trước full:
`cfb870e9796d9e20d01c8142b1bf4e397c154923`.

Raw run `framework_v3_full_20261019T162924Z/` COMPLETED; independent audit
`framework_v3_audit_20261019T165126Z/` PASS. Ba trace post hoc nằm trong
`framework_v3_diagnostic_{scratch_ppo,warm_bc,teacher_ppo}_20261019T1652*Z/`.
Các thư mục raw và checkpoint được giữ ở lab, không đưa vào Git.

## Thiết kế và kiểm chứng

Giữ đúng kernel, objective H48+tail16, N4/K2 anchor và bốn case v1/v2, 56
feature observation, 21 legal partial matching/WAIT, mask feasibility và
MaskablePPO MLP [64,64]. Teacher rollout nhìn current state cùng forecast
independent 6 tick ×16 scenario, không nhìn realized physical tape. Mỗi
replica lấy 32 train-only teacher episodes; 96 episode/4.608 label và
1.082.304 forecast transitions. BC actor 10 epoch/replica, 180 updates tổng;
train NLL giảm từ 0,759→0,437; 0,742→0,388; 0,737→0,395. Mỗi scratch/teacher
arm có 199.680 PPO decisions/replica; tổng 1.198.080. Initial weights và
physical training seed sequences giống nhau trong từng cặp; action làm
trajectory khác nhau. Không tính teacher data là sample miễn phí.

DEV chọn riêng scratch trong 12 checkpoint, teacher PPO trong 12 checkpoint
và warm BC trong ba checkpoint bằng 16 anchor seeds, 432 episode. Chọn scratch
seed 64000001/step199680 (DEV 805,25); teacher seed 64000000/step199680
(781,38); warm seed 64000000/step0 (785,25). Sau selection mới mở 64 TEST
seeds/case, tổng 1.536 episodes cho sáu controller. Full wall 1.314 giây,
gồm khoảng 70 giây xây teacher bank, ~1.017 giây PPO hai arm cộng lại,
432 DEV, 1.536 TEST. Teacher-bank selection tiêu 1.082.304 simulator
transitions; rollout inference trên anchor thêm trung bình 11.238 forecast
transitions/episode, còn learned-policy inference không rollout.

Audit runner và verifier độc lập kiểm tra source/artifact/checkpoint hash,
teacher label legality và physical replay, split/grid, initial weights và
paired training tapes, BC/PPO update counts, DEV-only selection, reward=−cost,
request conservation, zero invalid assignment, checkpoint reload và toàn bộ
paired bootstrap intervals. Feasibility được **mask/simulator bảo đảm**;
không coi đó là kỹ năng policy tự học. 52 targeted tests và Ubuntu smoke đã
PASS trước commit/full.

## TEST mới và hai contrast đồng chính

Cost H+tail thấp hơn tốt hơn. Hai CI hai phía 97,5% với 20.000 paired
seed-bootstrap draws là tiêu chí đã đăng ký; cả mean và cận trên phải <0.

| Contrast anchor | Mean | CI97,5% | Episode win/tie/loss | Quyết định |
|---|---:|---:|---:|---|
| Teacher PPO − scratch PPO | −44,92 | [−62,06; −28,73] | 34/27/3 | H1 đạt |
| Teacher PPO − warm BC | −6,58 | [−25,27; +4,89] | 7/49/8 | H2 chưa đạt |

Điểm ước lượng teacher thấp hơn warm, nhưng CI chứa 0. Hai giả thuyết cùng
đạt là điều kiện kết luận joint success, nên kết quả joint âm tính. Scratch
và teacher được ghép initial weights/training physical streams theo replica;
DEV chọn hai replica khác nhau, vì vậy TEST so quy trình selection được khóa,
không phải so hai checkpoint cùng seed riêng lẻ. CI bootstrap trên physical
seeds **không** bao phủ training-seed/selection uncertainty đầy đủ.

| TEST case (64 paired seeds) | Risk | Rollout | Scratch PPO | Warm BC | Teacher PPO | Teacher−scratch | Teacher−warm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reciprocal scale2, anchor | 802,72 | 703,72 | 814,30 | 775,95 | 769,38 | −44,92 | −6,58 |
| Interchangeable scale1 | 753,84 | 687,52 | 815,31 | 763,52 | 763,05 | −52,27 | −0,47 |
| Reciprocal scale3 | 993,41 | 1.039,77 | 986,75 | 955,52 | 945,38 | −41,38 | −10,14 |
| Specialist/generalist scale2 | 1.009,64 | 976,83 | 1.043,36 | 1.011,63 | 1.028,30 | −15,06 | +16,67 |

Ba case ngoài anchor chỉ là transfer diagnostics, không phải co-primary
confirmation. Trên anchor teacher thấp hơn risk 33,34 (CI95% mô tả
[−56,64;−13,09]) nhưng vẫn cao hơn rollout 65,66 ([+51,62;+79,78]).
Interchangeable teacher cao hơn risk 9,20 ([+0,16;+18,83]) và rollout 75,53;
specialist cao hơn risk 18,66 ([+2,33;+34,78]) và rollout 51,47. Scale3
teacher tốt hơn risk 48,03 và rollout 94,39 trên panel này, nhưng không được
chọn/chỉnh theo scale3. Không có ưu thế nhất quán trước strong baselines.

Sức khỏe anchor: H availability teacher 60,16%, warm 59,71%, scratch 60,24%,
risk 57,69%, rollout 59,64%; failures tương ứng 4,66/4,84/7,88/5,39/1,83.
Teacher cost tốt hơn scratch chủ yếu đi với ít failure cost hơn (93,05 so
137,58) và maintenance cost thấp hơn (57,30 so 60,69), trong khi H
availability không hơn scratch. Không gọi đây là cải thiện đồng đều mọi
metric. Specialist teacher availability 47,00% và queued CM 17,72 ticks,
kém warm 48,68% và 14,11 ticks.

## Mechanism post hoc và giới hạn

Theo plan riêng `framework_v3_diagnostic_plan.md`, trace ba checkpoint trên
chính panel TEST đã mở; chỉ diễn giải mô tả. Anchor scratch chọn matching có
thể mở rộng ở 12,14% tick, teacher 2,47%, warm 1,69%; teacher thực hiện ~0,711
PM start/tick so scratch 0,641. Teacher đồng ý risk ở 96,97% tick so scratch
87,11%; một số khác biệt hiếm vẫn đổi outcome dài hạn. Teacher và warm cho
kết quả gần nhau: paired episode cost bằng nhau ở 49/64 seed anchor. Trên
specialist, teacher WAIT dù còn việc legal ở 12,47% tick, so warm 7,29%; CM
queue exposure tăng, đi cùng cost +16,67 so warm. Vì trajectories khác sau
action, trace không tự chứng minh quyết định cụ thể gây chênh lệch cost.

Kết quả hỗ trợ *teacher assistance* với extra planner computation và train
labels, chưa chứng minh RL improvement beyond imitation. PPO có thể làm giảm
quality trên skill graph khác. Fixed 21-action catalogue và flat 56-feature
MLP chưa biểu diễn edge opportunity cost một cách cấu trúc; BC loss thấp trên
teacher states chưa đảm bảo hành vi khi policy tự đi sang states khác. Bốn
checkpoint và 199.680 steps/seed chưa tách hết optimization, delayed credit,
6-tick teacher horizon và representation. Không suy ra rollout optimal;
scale3 rollout cũng thua risk. Simulator synthetic chưa có routing sản xuất,
data nhà máy, RL gửi request hay quy mô N/K biến đổi. Chưa có bằng chứng để tuyên bố novelty, optimality hoặc áp dụng thực tế.

## Tái lập trên Ubuntu lab

```bash
cd /home/bachng/ht-pdm-fjsp
uv sync --frozen
uv run pytest -q tests/test_maintenance_framework_v1.py tests/test_maintenance_framework_v2.py tests/test_maintenance_framework_v3.py tests/test_maintenance_dispatch.py tests/test_maintenance_requests.py tests/test_maintenance_request_experiment.py
RUN_ID="framework_v3_replica_$(date -u +%Y%m%dT%H%M%SZ)"
uv run python -m ht_pdm_fjsp.maintenance_framework_v3_campaign --profile full --device cpu --output-dir "research_result/$RUN_ID"
AUDIT_ID="framework_v3_audit_$(date -u +%Y%m%dT%H%M%SZ)"
uv run python research_result/verify_framework_v3.py --run-dir "research_result/$RUN_ID" --output-dir "research_result/$AUDIT_ID"
```

Chạy trên commit `cfb870e` để lấy đúng source v3; chọn run ID mới, giữ
`research_result/` và checkpoint raw. Các nguồn v1/v2 vẫn cần khi tái lập
riêng hai vòng trước. Artifact manifest chứa exact config/seeds/source hashes.
