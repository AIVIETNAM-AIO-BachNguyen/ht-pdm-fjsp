# Stochastic-policy credit: critic hiện tại ổn, chưa đủ căn cứ thử GAE

Đã chạy đủ 21 frozen checkpoint ×64 stochastic episodes trên panel DEV mới:
1.344 episodes,64.512 policy decisions,80.640 physical ticks. COMPLETED/audit
PASS trong30,000585 giây, không update, teacher labels hoặc scientific tests.
Protocol eb87dd5, code557f401,9testsPASS2,82s rồi commit/push trước chạy trên lab.

Artifact:
`technician_capacity_onpolicy_credit_diagnostic_v1_20261016T074940212584752Z`.
Hai source RSC519bindings/factored548bindings, mọi checkpoint và weights giữ nguyên.
Seed mới240100000..63; action240200000..2,67raw/70derived được audit. Cùng physical
tapes giữa21models, RNG core ghép theo replica index; PPO/norm dùng cùng role riêng.
Checkpoint và initialization giữa hai source khác nhau; không coi cross-source
comparison là causal effect của factorization.

## Kết quả candidate, giữ đủ ba replica

D là độ phân tán bốn block gradient / bình phương norm của gradient trung bình.
Các advantage được chuẩn hóa toàn batch đúng PPO; gradient là actor score
surrogate tại behavior weights, không chứa entropy/value, không có optimizer.

| Source / train seed | Critic EV | Abs mean error / return SD | GAE D / MC D | Cosine mean gradient | Đủ gate? |
|---|---:|---:|---:|---:|---|
| RSC220000000 |0,98099|0,03180|0,37452|0,9999|Có|
| RSC220000001 |0,98438|0,00471|0,95298|1,0000|Không|
| RSC220000002 |0,98212|0,02389|0,78960|0,8494|Không|
| Factored230000000 |0,95166|0,04395|0,18758|0,5240|Không|
| Factored230000001 |0,93558|0,02514|1,00212|0,9903|Không|
| Factored230000002 |0,94091|0,08090|0,15561|0,7864|Không|

Điều kiện đã khóa yêu cầu MỖI candidate replica đạt EV≥0,5, abs mean error/SD≤0,1,
D-ratio≤0,75, cosine≥0,8 và mean MC gradient norm²>1e-20. Cả hai source family
đều không đạt; preferred_followup=null. Không mở lượt train GAE tự động.

GAE.95 giảm variance advantage THÔ ở cả6candidate (ratio≈0,306..0,417), nhưng
điều đó không đủ: PPO chuẩn hóa advantage, và các score gradient tương tác với
chúng. Các MC D của RSC lần lượt16,484 /0,511 /97,209; GAE6,174 /0,487 /76,756.
Ở factored, hai replica thay hướng gradient khá nhiều. Đây là mô tả finite-batch
trên4blocks, không phải chứng minh bias hay variance trên quần thể.

Cross-fitted time-only residual skill của RSC≈0,135..0,241 và factored≈0,003..0,101.
EV toàn bộ cao một phần vì remaining-time giải thích nhiều return; không diễn
giải EV≈0,98 thành giá trị hoàn hảo trên từng state/action. Tuy nhiên calibration
stochastic hiện tại không cho thấy lỗi value-scale rõ ràng như vòng completion
v1 cũ. Chưa đo early-training calibration, không loại mọi vấn đề credit learning.

Các stochastic cost/máy chỉ là chẩn đoán, không thay greedy FULL cũ:
RSC candidate272,418 /270,191 /272,020; factored309,004 /313,512 /315,793.
PPO261,313 /262,324 /261,840; norm261,527 /261,863 /262,020.
Không reselect checkpoints, không đổi verdict2/16 hoặc7/15 trước đây và khôngCI.

## Kiểm chứng và hệ quả nghiên cứu

Mỗi episode khớp reward với extendedcost,48decision+12tail,0invalid. Behavior
log/value replay nằm trong tolerance gốc; MC float64 khớp learner float32≤5e-4,
GAE lambda1 khớp G−V≤1e-10, parameter digests và grad fields không đổi.

Verifier độc lập e1274a3: PASS134diagnostic bindings+519/548source bindings,
đủ21models/1.344episodes/64.512decisions; tính lại MC/GAE bằng tổng tường minh,
episodecost, legality, shared populations,168storedgradientblocks và fixed gate.
Artifact `technician_capacity_onpolicy_credit_verification_20261016T075510155077970Z`.
Đây là kiểm độc lập dữ liệu/phép tính; không tính lại neural gradients hoặc
physical trajectories bằng implementation khác.

Kết quả không nói GAE vô dụng nói chung; chỉ không đủ động cơ theo gate đã khóa.
Giữ PPO-MC và normalized critic nếu nghiên cứu cơ chế policy tiếp. Không chuyển
sang lambda khác để làm gate này qua, không kéo dài budget hoặc chỉnh encoder
tự động. Contribution vẫn phải là phối hợp với ablation cùng thông tin/budget.

GAE là prior art [Schulman et al.](https://arxiv.org/abs/1506.02438), đã thuộc
họ kỹ thuật dùng với [PPO](https://arxiv.org/abs/1707.06347); không phải đổi mới
của nghiên cứu này. Scientific goal vẫn chưa đạt, tests và larger-N còn đóng.

Copy thủ công về Mac khi cần; không có transfer tự động:

```bash
mkdir -p "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results" &&
rsync -avhP --partial \
  bachng@100.111.83.52:~/ht-pdm-fjsp/artifacts/ \
  "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/"
```
