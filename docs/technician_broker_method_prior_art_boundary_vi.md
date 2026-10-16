# Ranh giới đóng góp của broker-marginal PPO

Ghi chú đối chiếu nguồn sơ cấp ngày17/10/2026, trong khi FULL learning-efficiency
đang chạy. Đây không phải systematic literature review hay bằng chứng “đầu tiên”.
Không sửa hypothesis, budget, cadence hoặc gate của lượt đang chạy.

## Những gì đã có tiền lệ

[CAPG, Fujita & Maeda2018](https://proceedings.mlr.press/v80/fujita18a.html)
khai thác phép clipping action để tích phân phần randomness không ảnh hưởng
execution. [MPG, Eisenach et al.2019](https://arxiv.org/html/1806.05134) đặt việc
marginalize action transformations trong một họ estimator và phân tích giảm
variance. Vì vậy identity conditional score/Fisher decomposition, Rao–Blackwell
interpretation và “tính xác suất action sau biến đổi” không phải theorem mới của
broker proposal. Kết quả trên estimator không đảm bảo finite-sample clipped-PPO
AUC hoặc maintenance cost sẽ tốt hơn.

[MAT, Wen et al.2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html)
đã dùng encoder-decoder và sequential action formulation trong cooperative MARL,
học online on-policy. Do đó factorize joint actions hoặc tránh catalogue toàn bộ
joint action không tự nó đủ làm novelty. Không suy diễn complexity statement của
paper thành bảo đảm wall-time cho mạng/broker của mình.

[COMA, Foerster et al.](https://arxiv.org/abs/1705.08926) dùng centralized Q critic
và counterfactual baseline tích phân action của một agent khi giữ actions khác.
Đó là credit assignment; broker-marginal hiện tại thay likelihood của cùng
physical execution, vẫn state-value critic/MC advantage, chưa phải một COMA critic.

[OMDPG, Yu et al.2025](https://arxiv.org/abs/2507.09989) mô tả optimal marginal Q,
grouped actors/partial parameter sharing và vấn đề baseline drift trong sequential
updates. Đây là kiểm tra ở mức abstract, chưa thẩm định toàn bộ proof/code của
paper. Không được dùng chữ “marginal” trong tên như bằng chứng phương pháp của
mình đã khác hoặc mới; cũng không gán cho paper một winner-only matching density
khi chưa tìm thấy điều đó trong nội dung đã đọc.

## Phần khác biệt thực sự đang được kiểm tra

Policy phát independent technician proposals U; deterministic fixed-priority
winner-only broker tạo matching S=T_s(U). Khi broker không retry và priority chỉ
phụ thuộc state, preimage mass của matching đã quan sát factorize:

    q(S|s) = product_(m,j in S) p_j(m|s)
             * product_(j idle) [p_j(WAIT|s)
                 + sum_(m,w in S; w outranks j at m) p_j(m|s)].

Thuật toán đã triển khai tính likelihood matching này để dùng trong joint PPO
ratio, không liệt kê mọi matching khi sample/replay/select. Đối chứng giữ cùng
actor/info/capacity/broker/entropy/optimizer/budget, chỉ dùng proposal likelihood
p(U). Đây là khác biệt thuật toán xác định được; nó chưa mặc nhiên là đóng góp
được xác nhận. Fixed quality broker không được tuyên bố novel hoặc learned.

Actor hiện tại vẫn thấy full current state, broadcast cho các technician; critic
cũng centralized. Do đó không tuyên bố partial-observation decentralized
execution hoặc giảm communication. Raw ordered state chưa permutation-equivariant;
size-specific polynomial architecture chưa chứng minh zero-shot generalization.
Reference liệt kê21matching trong frozen-mode/H4diagnostics không phải scalable
implementation. H4oracle values không được đưa vào learner.

## Bằng chứng còn thiếu

FULL983040steps đã FAIL practical mechanism margin; frozen-mode và H4remedies
cũng không đủ động cơ đã khóa. Lượt learning-efficiency mới chỉ kiểm một câu hỏi
khác trên fresh random starts, với AUC/fixed-final endpoint. Dù DEV pass, vẫn cần
independent verification, sealed confirmation/CI, N/Kscaling với chi phí quyết
định thực, same-info trained ablation và health bounds. Muốn lập luận novelty
mạnh còn cần đối chiếu implementation/baselines gần nhất về sequential policies
và action transformations; không dùng thắng MAPPO local hoặc một theorem đã có
để thay thế các yêu cầu này. Hiện chưa có cơ sở tuyên bố thuật toán mới đã thành công.
