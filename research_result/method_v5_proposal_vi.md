# Method proposal v5: chấm toàn bộ matching qua năng lực còn lại sau quyết định

Viết trước implementation, DEV/TEST v5. Đây là một **follow-up có giới hạn**
sau v4, không sửa v4 hoặc diễn giải lại H1 v4. Kết quả v4 đã biết và ghi ở
`research_result/framework_v4_findings_vi.md`: temporal age2 thất bại ở
primary, additive queue price có secondary pooled improvement nhưng tắt
nhánh cùng weights chỉ đổi 83/12.288 action. Không claim novelty.

## Vấn đề cụ thể

Score v4 cộng các cạnh và trừ giá cạnh riêng lẻ. Một request khác có thể
được phục vụ bởi hai technician; hoặc cùng một matching có thể chiếm cả hai.
Giá cộng từng cạnh không tính chính xác năng lực còn lại *sau khi chọn cả
matching*. Nó cũng không xem min earliest start của queued request dưới
matching, nên một tác động thời gian trực tiếp có thể chìm trong base scores.
Đây là giả thuyết giải thích, chưa được v4 chứng minh.

## Ý tưởng và cơ chế

Centralized scratch RL vẫn quyết định matching hoặc WAIT ở đầu mỗi tick H48,
với state, request lifecycle, reward H48+tail16 và feasibility mask như v4.
Một shared edge network tạo base benefit cho (m,j). Một shared nonnegative
potential q(m) học từ state hiện tại của *queued* request. Với mỗi action a
trong catalogue 21 matching, tính worker busy time dự kiến r(j,a): thời gian
remaining nếu đã bận, duration d(m,j) nếu a gán j cho m, hoặc 0 nếu trống.
Cho request m còn queued nhưng không được a phục vụ, ước tính sớm nhất có
technician phù hợp là min r(j,a) theo eligible j. So với cùng đại lượng trước
a, chênh lệch không âm Δwait(m,a) tính bằng **tick thật**, không phải một
forecast sample. Actor logit:

`sum_(m,j)∈a base(m,j) − softplus(scale) × sum_(m queued, m not served by a) q(m) × Δwait(m,a)`.

Đây là *post-decision capacity feature* cho policy, không phải reward
shaping, planner, hoặc đọc realized future. Nó phụ thuộc cả matching:
serving một request loại nó khỏi externality; chiếm cả hai worker có thể
làm min availability của request khác tăng. Không giả định min availability
là completion-time forecast chính xác vì arrivals/failures tiếp diễn.
Không thêm healthy age2 gate trong v5; v4 đã bác bỏ thiết kế cụ thể đó.

Ba actor structured `edge_only`, `additive_queue` (công thức v4),
`postdecision` dùng cùng machine/worker/edge/potential modules, optimizer,
initial parameter values theo training seed và physical/case training tape.
`flat` PPO được train từ đầu cùng budget. Không IL/pretraining. So rule
FCFS, risk matching và bounded rollout dùng cùng current-information
contract, physics, objective và TEST seeds. Feasibility vẫn do mask.

## Giả thuyết bác bỏ được

H1 primary: postdecision giảm case-balanced H+tail16 cost so additive_queue
trên bốn case N4/K2 cố định. DEV-only checkpoint selection, 64 *mới*
physical TEST seeds/case, paired cluster bootstrap 20.000 draws theo seed;
confirm nếu mean<0 và upper CI95%<0. H2 secondary: postdecision so
edge-only, flat, risk, FCFS, rollout; case-level, health, latency và
failure/queue metrics là descriptive (không hiệu chỉnh multiple comparisons).
Nếu H1 fail, không thay scale/architecture/horizon theo TEST. Nếu H1 dương
nhưng không thắng risk/rollout, claim chỉ method contrast; không claim
controller superior. Nếu selected checkpoint có cost benefit nhưng read-only
same-weight penalty ablation gần như không đổi action, chỉ có thể nói
inductive bias/optimization khác, không gán trực tiếp cho online residual
capacity. Kiểm tra permutation equivariance actor tại N4/K2 và request/
resource invariants; không claim transfer N/K.

## Prior art và ranh giới

Learning to Dispatch (Zhang et al., NeurIPS 2020) và DANIEL đã học dispatch
scores/cạnh từ cấu trúc; online bipartite matching đã xét skip và demand
uncertainty. Maintenance RL và bounded rollout đã xét quyết định theo thời
gian. V5 không tuyên bố phát minh post-decision state hay scarce-resource
pricing. Câu hỏi mới trong **simulator hiện tại** là: actor action-conditioned
residual waiting có thêm giá trị so additive queue price khi tất cả còn lại
được kiểm soát hay không. Muốn claim novelty phải đối chiếu full literature
và benchmark ngoài simulator này.
