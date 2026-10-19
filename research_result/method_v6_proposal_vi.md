# Method proposal v6: giá trị của parameter sharing trên cạnh machine–technician

Viết trước implementation, DEV/TEST v6. V4/v5 đã loại bỏ hai cơ chế capacity
được đề xuất (age2 anticipation và post-decision residual wait) ở primary;
additive queue benefit v4 không lặp rõ v5. Nhưng family chấm action bằng
shared pair scores thắng flat MLP trong v5 và không rõ trong v4. Câu hỏi còn
mở là chính **sharing cùng một edge scorer** có giúp học điều phối, hay kết
quả do action factorization/optimizer/checkpoint/physical panel. Đây là một
method component cũ trong literature, không là novelty claim.

## Thiết kế và cơ chế

Cùng centralized RL, H48+tail16, N4/K2, 56-d current observation, 21
partial matchings gồm WAIT, legal mask, exogenous threshold requests và
four-case uniform training mix. Actor `shared_edge` dùng chung machine encoder,
worker encoder và một edge MLP 40→32→1 cho mọi (m,j); matching logit là tổng
score cạnh. Actor `unshared_edge` dùng cùng machine/worker encoders, cùng
matching-sum decoder nhưng tám edge MLP 40→4→1 độc lập cho các vị trí nhãn.
Hai edge-head constructions có 1.345 vs 1.352 parameters (khác 7), cùng
critic/PPO và information. Đây cô lập parameter tying ở edge head; common
encoders vẫn chia sẻ và không được quy mọi effect cho toàn bộ representation.
`flat` 56→MLP→21 matching logits là architecture control. Không dùng capacity
penalty, imitation hay teacher. Feasibility từ mask.

Shared actor là permutation-equivariant về logits khi relabel machines,
technicians và actions. Unshared actor không bị ràng buộc equivariant. Kiểm
tra state-level exact remapping và true-state replay là structural QA,
không được gọi là generalization N/K. Không chọn permutation hoặc case theo
TEST. Direct quality claim vẫn là cost trên bốn original-label cases.

## Hypothesis và đối chứng

Primary H1: shared_edge giảm case-balanced H+tail cost so unshared_edge khi
cùng train budget, matched physical tapes, DEV selection và sealed TEST mới.
Năm training seeds (thay vì chỉ một selected checkpoint), mỗi arm 100.352
PPO steps với checkpoints ở 50.176 và 100.352. DEV 16 seeds/case chọn một
checkpoint **trong mỗi training seed/arm**, không chọn best training seed.
TEST 64 fresh physical seeds/case cho đủ 5 model pairs. Mỗi training-seed ×
physical-seed cell trung bình đều bốn case. Hierarchical paired bootstrap
20.000 draws resample 5 training seeds và 64 physical seeds; confirm nếu
mean<0 và upper two-sided CI95%<0. Đây là tiêu chuẩn cố ý tính training
variation, hạn chế của v4/v5 selection-single-model.

Secondary descriptive: shared vs flat/risk/FCFS/rollout, unshared vs flat,
case-level contrast, failure/queue/availability, selection/training compute,
parameter counts, actor permutation. Baselines dùng cùng physics/current
information/TEST seeds; rollout forecast độc lập. Báo đủ wins/losses trên
train seeds và physical seeds, không gán thắng lợi cho shared nếu chỉ hơn flat
mà không hơn unshared. Nếu primary fail, không đổi model widths hoặc tăng
budget sau TEST. Một FULL duy nhất, hard stop 4h, không adaptive sweep.

## Literature và giới hạn

Learning to Dispatch (NeurIPS 2020) dùng graph RL học dispatch rule từ
scratch và transfer kích thước; HetGPO/AirME (ICAPS 2022) đã dùng graph RL
cho shared maintenance crews và composite/skip decisions; DANIEL và online
matching cũng khai thác relational/edge structure. V6 chỉ kiểm chứng lợi
ích *parameter tying của một shared edge head cụ thể* so cùng factorized
matching decoder trong simulator hiện tại. Không claim phát minh edge RL,
permutation equivariance, optimality, scalable graph policy hoặc ứng dụng
thực địa. Nếu có giá trị, nó là evidence phương pháp có giới hạn; cần
benchmark ngoài N4/K2 và literature comparison sâu hơn trước paper claim.
