# Phân tích online TD + train-only assignment — 04/10/2026

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent, validate.
- Verification Status: **ANALYZED**, kèm replay checkpoints và exact-cost arithmetic; không retrain.
- Protocol: `ra_qmix_td_assignment_v1`; locked plan và source tại commit 5799a05.
- Source: `lab_results/ra_qmix_td_assignment_full_cpu_20261004T053816Z/`.
- Source commit: `5799a0527796caea511edf281700256e2e412a6e`, clean theo manifest.
- Origin Date: 2026-10-04, Asia/Taipei; Version Label: td_assignment_analysis_v1.
- Derived script/checks: `reports/ra_qmix_td_assignment_analysis_20261004T121900Z/`.

## Kết luận

**TD + assignment đạt exact oracle cost ở cả ba cells và mọi training seed, học đúng toàn bộ training assignments, không nominal regression.** Assignment supervision hoạt động được cùng online TD trong diagnostic này. Tuy nhiên TD-only cũng đã khá gần oracle, nên benefit theo deployed cost nhỏ hơn practical thresholds đã khóa: **primary FAIL**, supervised control **PASS**, `empirical_attribution_supported=false` theo protocol.

FAIL không có nghĩa supervision không học được hoặc cần thay mixer. Đây là FAIL của minimum practical improvement gate; treatment đã chạm oracle floor. Không đổi gate hậu nghiệm để biến result thành PASS. Kết quả descriptive hỗ trợ tính khả thi của TD + supervision trong finite known cells, chưa chứng minh utility đủ lớn theo endpoint đã khóa hoặc một thuật toán thực tế không cần oracle teacher.

## Verification

Run COMPLETED từ 13:38 đến 15:45 ngày 04/10/2026 giờ Đài Bắc, khoảng 2 giờ 8 phút. Đủ 90 checkpoints, 3.600.000 training steps, 600.000 training episodes, 1.338.570 updates, 13.410 finite progress rows, 4.500 unique evaluation episodes và 40.200 state-model rows. Mỗi model 14.873 updates; 60 RL models có 60.000 steps/10.000 episodes, 30 controls không có training interaction. Target sync đúng 29 lần/RL model. Mọi output trong manifest hiện diện.

Analysis đã kiểm tra:

- 600.000 training rows: episode counters liên tiếp và duy nhất, env steps = episode × 6, invalid requests = 0.
- Evaluation seeds đúng 98200–98249; unique keys; objective khớp tổng cost components.
- Teacher batch hashes khớp giữa ba arms trong mỗi cell/seed.
- 1.068 teacher assignments hợp lệ, tối ưu và chỉ thuộc train states; held-out teacher fields để trống.
- Replay **90 checkpoints/40.200 quyết định**, khớp stored actions.
- Tái dựng finite oracle bằng source implementation, tính lại discounted/undiscounted costs của **90 models**, khớp CSV.
- Primary/control gates và paired metrics tính lại từ raw model CSV, khớp summary.

Đây là arithmetic consistency check với cùng oracle implementation, không là implementation độc lập hoặc training reproduction. Training reward reconciliation được assert trong runner; từng transition/reward không được lưu đầy đủ nên analysis không tái kiểm tra tất cả training rewards. Sealed 201–300 đóng; freshness chỉ trong registry 170 prior local manifests.

CF = 0, ID + peer và original monotonic mixer giữ nguyên. TD rewards chia fixed 20 cho cả hai RL arms; CE weight = 1, auxiliary batches lấy uniform training states độc lập với replay. Same initialization/cadence/teacher draws không có nghĩa replay hay trajectories giống nhau. Teacher dùng exact optimal actions dựa trên future dynamics, vẫn là privileged supervision.

## Primary practical gate

Equal weighting shared-preference/pressure exact undiscounted initial-policy cost, paired 10 training seeds:

| Endpoint | Kết quả | Ngưỡng khóa | Gate |
|---|---:|---:|---|
| Absolute reduction | 1,657148 | ≥2 | FAIL |
| Relative reduction | 6,53% | ≥10% | FAIL |
| Improving seeds | 8/10, hai ties | ≥8/10 | PASS |
| Nominal cost change | −6,98% | ≤+10% | PASS |

Baseline mean cost **25,384648**, treatment **23,727500**. Paired delta **−1,657148**, CI95% **[−3,125882; −0,188414]**. CI là supportive theo locked Student-t protocol; khoảng toàn negative không thay minimum-effect gate. Không suy zero effect/equivalence từ primary FAIL.

**Posthoc headroom check:** oracle mean của hai diagnostic cells là (4 + 43,455)/2 = **23,7275**. So với observed TD baseline, maximum achievable gain của bất kỳ policy nào cũng chỉ **1,657148 / 6,53%**. Treatment đạt maximum đó. Vì vậy thresholds 2 / 10% không thể đạt với baseline cụ thể của run này, ngay cả optimal treatment. Đây là giới hạn sensitivity của benchmark/endpoint trong run này, không là lý do sửa gate sau khi thấy outcome.

## Performance và seed tails

Exact expected **undiscounted cost từ initial state**, mean 10 seeds; thấp hơn tốt hơn:

| Policy | Nominal | Shared preference | Pressure |
|---|---:|---:|---:|
| Oracle | 4,000 | 4,000 | 43,455 |
| TD-only | 4,300 | 4,650 | 46,119 |
| TD + assignment | **4,000** | **4,000** | **43,455** |
| CE-only control | **4,000** | **4,000** | **43,455** |

Treatment và control đạt oracle cost ở **mọi 30 cell × seed models mỗi arm**. TD-only đạt oracle ở 5/10 nominal, 3/10 shared và 5/10 pressure seeds. Không seed nào có diagnostic treatment cost cao hơn TD-only.

Pressure TD-only seeds 98006/98009 có costs 53,205 / 53,683875; chúng đóng góp paired diagnostic deltas −5,125 / −5,614438. Improvement không đồng đều; tail seeds ảnh hưởng mean/CI rõ rệt.

Pure TD nominal/shared có zero rollout failures nhưng vẫn collisions/waiting: nominal means 0,1 / 0,6; shared 0,4 / 1,1. Treatment/control có collisions và waiting = 0 ở cả ba cells. Pressure optimal policy vẫn có stochastic failures: treatment/control mean 1,64, TD-only 1,706.

Pressure sampled objective TD-only 48,738 so với exact 46,119; treatment/control 46,72 so với exact 43,455. Không thay exact primary metric bằng mean của 50 sampled episodes/model. Treatment/control có cùng sampled results từ initial state ở evaluation panel này; không suy policies trùng ở mọi states.

## Teacher fit và phân vai

Control gate yêu cầu mỗi cell train regret ≤0,10 và ≥8/10 seeds có ≥99% optimal training decisions. Control đạt **regret 0, 100% optimal, 10/10 seeds mỗi cell**: PASS. TD + assignment cũng đạt 100% training teacher agreement và train regret 0. Chưa thấy TD ngăn supervision fitting trong budget này.

| Cell | TD-only train regret | TD + assignment train regret | TD-only teacher-heldout regret | TD + assignment teacher-heldout regret | Control teacher-heldout regret |
|---|---:|---:|---:|---:|---:|
| Nominal | 0,284575 | 0 | 0,329128 | **0,002264** | 0,013690 |
| Shared preference | 1,079485 | 0 | 0,680186 | **0,000241** | 0,110418 |
| Pressure | 0,595602 | 0 | 0,538528 | **0,045923** | 0,147662 |

Treatment teacher-heldout optimal fractions: **98,70% / 99,74% / 99,32%**, so với TD-only 75,71% / 71,43% / 80,76%. Uniform state regret và trajectory cost khác nhau: pure TD có nhiều lỗi off-trajectory nhưng vẫn gần oracle từ initial state; treatment/control có một ít held-out lỗi vẫn đạt initial cost oracle.

Trên original-identical-local-input states cần unequal oracle actions, tổng state × seed pairs:

| Cell | TD-only optimal | Treatment optimal | Control optimal |
|---|---:|---:|---:|
| Shared preference | 49/130 | **130/130** | **130/130** |
| Pressure | 64/110 | **110/110** | **110/110** |

Có cải thiện cụ thể về phân vai tại symmetry states.

Late updates 14000–14873: TD loss baseline → treatment là nominal 0,031608 → 0,039587; shared 0,029496 → 0,037664; pressure 0,104424 → 0,101679. Treatment CE khoảng 0,00015–0,00033. TD loss nhỏ hơn không bảo đảm policy tốt hơn; các means này không chứng minh gradient conflict, convergence hoặc mediation.

## Held-out có nghĩa gì?

Teacher constraints/actions chỉ dùng training states. Nhưng RL đã gặp và học TD rewards ở nhiều teacher-heldout states:

| Cell | Tổng teacher-heldout states | TD-only mean visited | Treatment mean visited |
|---|---:|---:|---:|
| Nominal | 77 | 68,1 | 69,0 |
| Shared preference | 77 | 69,5 | 71,6 |
| Pressure | 118 | 93,5 | 92,8 |

Vì vậy held-out regret của RL không là generalization sang unseen states/environments. CE-only control không có environmental training nên held-out của nó đo teacher-only transfer; treatment có thêm information từ real rewards. Lower treatment held-out regret so với control là secondary observation, chưa là một riêng biệt superiority claim.

## Ý nghĩa cho bước tiếp theo

Giữ shared agent/mixer là candidate hợp lý: TD và assignment loss cùng hoạt động tốt với fixed ID + peer trong các cells này. Supervision xóa oracle cost gap và cải thiện decision coverage, nhưng practical gate chưa đạt vì observed baseline đã gần floor. Chưa chứng minh original adaptation/scale problem được sửa hay teacher đủ rẻ để triển khai.

Nếu tiếp tục, ưu tiên protocol trên fresh configs/larger systems có measurable headroom và teacher không cần full oracle, chẳng hạn consistent assignment heuristic hoặc teacher từ training data. Giữ same-information TD baseline, nominal guard và seed-tail reporting. Khóa benchmark/seeds/teacher quality trước outcomes; không hạ thresholds hoặc chọn hard cells hậu nghiệm rồi gọi confirmatory. Analysis này không chạy experiment mới.

## Statistical limits và fallacy scan

Inference unit là **10 paired training seeds**, không phải 600.000 training episodes, 4.500 evaluation episodes hay 40.200 state rows. Một teacher/split mỗi cell; seed CI không bao gồm teacher/split/environment variability. Student-t CI supportive, n nhỏ, normality chưa được kiểm chứng. Per-cell, regret, mechanism và tail comparisons là secondary.

Coverage: **11/11 fallacy types checked**.

| Loại | Kiểm tra/giới hạn |
|---|---|
| Simpson's paradox | Report cells riêng; uniform regret và trajectory cost tách rõ. |
| Ecological fallacy | Mean không đại diện mọi seeds; pressure tails và finite population được nêu. |
| Berkson's paradox | Không lọc models/cells thắng; critical subset chỉ là mechanism breakdown. |
| Collider bias | Primary không conditioning vào failures/collisions hậu training. |
| Base rate neglect | Pressure có failure law và irreducible cost khác; optimal policy vẫn có failures. |
| Regression to mean | Same-run paired controls; không dùng prior MSE run làm causal baseline. |
| Survivorship bias | Đủ 90 models và 600.000 training rows; không bỏ/retry seeds. |
| Look-elsewhere effect | Primary FAIL giữ nguyên; gap removal/regret không thay endpoint. |
| Garden of forking paths | Budget/teacher/threshold khóa; headroom check đánh dấu posthoc. |
| Correlation vs causation | Controlled intervention, nhưng practical gate FAIL; không suy unique cause của adaptation failure. |
| Reverse causality | Objectives trước outcomes; late losses không là mediation proof. |

Overall: **primary FAIL / control PASS**; protocol empirical attribution chưa được hỗ trợ. Descriptive oracle costs/teacher fit mạnh nhưng có phạm vi. Raw artifacts giữ nguyên; arithmetic và checkpoint decisions checked, training reproduction chưa chạy.
