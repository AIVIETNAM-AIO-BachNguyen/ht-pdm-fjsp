# Ledger nghiên cứu Reservation Framework

Ngày mở: 2026-10-18. Goal A/B/C còn **ACTIVE**. Kết quả lịch sử không được tái dùng làm xác nhận mới.

| Claim | Bằng chứng hiện có | Bằng chứng còn thiếu |
|---|---|---|
| A: teacher N4 giúp student thích nghi | V4 dùng teacher PPO N4, 122.880 trạng thái/bundle, distillation và PPO riêng từng N; cost/máy V4 N4 262,1729, N8 265,9545, N16 266,5710, N32 264,8450 | Scratch cùng kiến trúc/reservation/RL budget trên panel mới; fixed-final, AUC, ổn định, health, zero-shot riêng; hạch toán compute đầy đủ |
| B: selected-edge prefix giúp ngoài mask | V4 full hơn no-reservation ở N8/N16/N32 (gain 20,79/43,75/56,86%, CI dương), N4 chỉ noninferior; hai cơ chế được thay đổi cùng lúc | Mask-only cùng action set/order/teacher/budget, trajectory audit; pooled gain ≥1% với cận dưới CI một phía >0 trên panel mới |
| C: chất lượng, health, scaling bảo trì | V4 xác nhận benchmark cân bằng K=N/2, horizon 48+12, zero invalid, `model.select` p95 1,527–9,460 ms; N32 failures cao hơn deadline dù trong biên V4. Shared-technician factorial/coupled follow-up cho thấy FCFS/risk mạnh và tác động duration/pending đáng kể | Hai họ cấu hình non-V4 khóa trước, FCFS/risk chọn bằng DEV, matched mask-only, health/latency end-to-end, audit độc lập, train riêng/fine-tune/zero-shot tách biệt |

## Vòng B1: prefix so với mask-only

Giả thuyết: thông tin embedding của matching đã chọn làm policy cùng reservation giảm cost so với chỉ mask trên N4/N8. Nghiệm vô hiệu: mask-only có cùng legal set, summary nguồn lực và thứ tự, nên gain full không vượt mask-only. Protocol: `docs/technician_prefix_mechanism_v1_plan.md`. Không dùng V4 TEST để xác nhận.

**Kết quả DEV (18-10-2026): FAIL, TEST sealed.** Source run:
`artifacts/technician_prefix_mechanism_v1_development_20261018T070951Z`,
code commit `ada86a338cda7d401d62e4611ecd63da553c199e` đã push trước FULL.
Run COMPLETED sau 4.490,05 s, 24 model, 184.320 training episode,
3.072 DEV episode; audit nội bộ PASS. Kiểm toán độc lập từ CSV, checkpoint
choices, source/output hashes tại
`artifacts/technician_prefix_mechanism_v1_development_independent_audit_20261018T120243Z`
PASS và xác nhận gate FAIL. Pooled full vs mask-only gain chỉ **+0,2205%**
(ngưỡng DEV +0,5%), N4 **+0,6746%**, N8 **−0,2437%**. Mỗi training
replica N4: +0,5437%, 0,0000%, +1,4890%; N8: +0,3504%, **−1,6699%**,
+0,5801%. Health point guards đạt; đây không cứu cost gate. Full vượt
no-reservation ở điểm trung bình N4/N8, nhưng contrast đó gộp chống trùng
và prefix, nên không chứng minh prefix hữu ích. Không mở confirmation B1.

Giới hạn: ba teacher/student replica, hai quy mô, 32 DEV episode/checkpoint;
đây là thất bại developmental của cơ chế B1 đã đăng ký, chưa là phủ định
confirmatory của mọi kiến trúc prefix. Bước kế tiếp: chẩn đoán trên train/DEV
đã mở xem prefix có thật sự đổi action sau khi reservation mask/resource
summary được cung cấp hay không; sau đó đăng ký tối đa một follow-up cơ chế
trên seed mới nếu có lý do. Không dùng TEST V4/B1 để chọn follow-up.

**Chẩn đoán B1 đã đăng ký sau FAIL (18-10-2026): COMPLETED, exploratory.**
`artifacts/technician_prefix_b1_diagnostic_full_20261018T120638Z` chạy 192
episode DEV, 9.216/18.432 kiểm tra legal mask N4/N8 bằng nhau chính xác;
TEST vẫn sealed. Khi giữ nguyên trọng số full nhưng tắt prefix, quyết định
đổi ở 21/4.608 state N4 và 621/4.608 state N8. Cost/máy full thấp hơn
cùng trọng số không prefix lần lượt 2,045% và 2,624% trên panel này.
Do đó giả thuyết chẩn đoán “prefix hiếm khi đổi hành động” không đúng, đặc
biệt ở N8. Đây là can thiệp cùng trọng số trên quỹ đạo đã mở, có distribution
shift và **không** thay thế contrast chính với mask-only được huấn luyện lại;
B1 DEV vẫn FAIL. Bài học: biểu diễn prefix hiện tại có tác động, nhưng
mask-only học lại có thể bù phần lớn lợi ích. Follow-up hợp lý cần kiểm tra
liệu biểu diễn quan hệ giữa từng ứng viên và các cạnh đã chọn có vượt đối
chứng học lại trên seed hoàn toàn mới; không diễn giải chênh lệch cùng trọng
số là claim B.

## Protocol và trạng thái khoa học

| Nhánh | Giả thuyết đã khóa / source commit của run | Artifact DEV hiện hành | Trạng thái tính đến 18-10-2026 17:35 UTC |
|---|---|---|---|
| A1 teacher→student | Teacher N4 + distillation giúp graph reservation trên N8/N16/N32 so với scratch cùng PPO budget; `a60f716` | `/home/bachng/ht-pdm-fjsp-teacher-transfer-v1/artifacts/technician_teacher_transfer_v1_development_20261018T134420Z` | RUNNING, training pool; TEST sealed. Verifier độc lập đã PASS trên smoke, chờ DEV hoàn chỉnh. |
| B2 relational prefix | Candidate-conditioned attention trên các selected edge giúp hơn mask-only học lại, sau B1 FAIL; `b2a271f` | `/home/bachng/ht-pdm-fjsp-relational-v2/artifacts/technician_relational_prefix_v2_development_20261018T134357Z` | DEV GATE FAIL; kiểm toán độc lập PASS; TEST sealed. |
| B3 selected-pair capacity prefix | Hai scalar tải dịch vụ và phục hồi yếu của cặp đã chọn, cộng B2 relational message; matched mask-only tắt cả hai; `a99ab1c` | `/home/bachng/.codex/worktrees/reservation-prefix-capacity-v3/ht-pdm-fjsp/artifacts/technician_prefix_capacity_v3_development_20261018T173438Z` | DEV GATE FAIL; kiểm toán độc lập PASS; TEST sealed. |
| C1 đa cấu hình | N8/N16 skill/K khác V4 và service 2–4 tick với request bền vững, đối chứng mask-only/scratch/FCFS/risk; `396ecb5` | `/home/bachng/ht-pdm-fjsp-diverse-v1/artifacts/technician_diverse_maintenance_v1_development_20261018T134420Z` | DEV_GATE_PASS với verifier độc lập PASS; confirmation mới đang chạy, chưa có verdict TEST. |

Các commit verifier/doc sau khi run bắt đầu không đổi mã learner/protocol/
registry được chụp trong artifact. Không diễn giải progress huấn luyện hoặc
smoke là bằng chứng đạt A/B/C. Nếu DEV thất bại, giữ TEST của đúng nhánh
niêm phong và ghi nguyên verdict trước follow-up.

**Chuẩn bị confirmation B2 (13:20 UTC):** nhánh riêng
`codex/reservation-relational-confirmation-v2`, commit `557bad1` đã push.
Runner từ chối DEV đã mở TEST hoặc snapshot mã học/physics, protocol,
seed registry khác DEV; kiểm tra trên snapshot DEV đang chạy PASS, 7 targeted
tests và smoke kỹ thuật PASS. Chỉ dùng nhánh này để mở B2 confirmation nếu
DEV gate và audit độc lập cùng PASS. Đây không phải kết quả khoa học B2.

**Audit trước TEST (13:33 UTC):** verifier A1 `a60f716` đã push, replay
feasibility trên cả quỹ đạo teacher và scratch; verifier B2 `1f6f687` đã push,
replay full và mask-only; verifier C1 `887723a` đã push, kiểm stage-level
full/mask-only trên tám seed TEST đầu mỗi cấu hình đã định trước và replay
mọi transition/request/cost trên toàn panel. Cả ba mở rộng verifier được
kiểm bằng smoke kỹ thuật PASS, không đổi learner/protocol/registry hay mở
TEST khoa học. C1 verifier phải báo rõ phạm vi kiểm stage là subset;
không được mô tả như kiểm mọi stage của mọi seed.

**Gián đoạn thực thi và recovery (13:41–13:44 UTC):** ba PID A1/B2/C1
biến mất cùng lúc, manifest cũ kẹt `RUNNING`, không có DEV summary hay TEST.
Phân loại ba artifact cũ là TECHNICAL_INCOMPLETE, không là verdict khoa học;
nguyên nhân/signal cụ thể chưa xác định. Incident hash-bound và các run DEV
mới dưới detached user service đã ghi tại
`docs/reservation_framework_execution_recovery_20261018.md`. Dùng artifact
mới để audit; không dùng partial quality/checkpoint cũ để chọn model.

**B2 DEV verdict (18-10-2026 17:12 UTC): FAIL, TEST sealed.** Run mới
hoàn tất sạch (`systemd` exit 0, manifest `COMPLETED`, audit nội bộ PASS),
không sử dụng artifact gián đoạn. Kiểm toán độc lập tại
`/home/bachng/ht-pdm-fjsp-relational-v2/artifacts/technician_relational_prefix_v2_development_independent_audit_20261018T171152Z`
PASS: khớp 260 source seals, 475 output bindings, 184.320 episode huấn
luyện, 3.072 episode DEV, 61.440 kiểm tra training tape ghép cặp, tính lại
checkpoint/cost/health và xác nhận `test_opened=false` (manifest SHA-256
`44f3ee1b272a16486e0e57a61031d368d84e9358aac36a1da49571b5a67e9c09`).
Gain full so với mask-only học lại theo thước đo đã đăng ký: pooled
**−0,1644%** (gate DEV +0,5%), N4 **−0,0168%**, N8 **−0,3119%**. Mọi
health point guard và zero-invalid đều đạt, nhưng không cứu được cost gate.
No-reservation có cost trung bình cao hơn full ở N8 DEV, song contrast đó
chỉ đo toàn dịch vụ
reservation/message; không chứng minh lợi ích của prefix. Không mở B2
confirmation hay đổi ngưỡng/panel. B1 và B2 là hai kết quả DEV âm riêng
biệt. Chẩn đoán/follow-up nếu có chỉ dùng train/DEV đã mở và seed mới được
khóa trước; không dùng B2 TEST.

**B2 chẩn đoán DEV cùng trọng số (18-10-2026 17:23 UTC):** kế hoạch
thăm dò được khóa sau verdict B2, 8 targeted tests và smoke 12 episode
PASS, mã `ea5f053` đã push trước full. Artifact
`/home/bachng/ht-pdm-fjsp-relational-v2/artifacts/technician_relational_prefix_v2_dev_diagnostic_20261018T171908Z`
COMPLETED, 192 episode DEV, `test_opened=false`; output bindings được
kiểm lại. Với checkpoint full đã chọn, chỉ tắt message mà giữ nguyên mọi
trọng số làm đổi 21/4.608 quyết định tick N4 và 549/4.608 tick N8;
legal mask bằng nhau ở 9.216 và 18.432 stage. Full có cost thấp hơn bản
tắt message cùng trọng số **1,3135% N4** và **1,2121% N8** trên DEV,
nhưng một replica N8 không có lợi. Đây là tín hiệu policy đã sử dụng
message, **không** là contrast full vs mask-only được huấn luyện độc lập;
verdict B2 DEV FAIL và TEST sealed giữ nguyên. B1 từng có cùng mô thức:
message đổi hành động nhưng không thắng matched mask-only. Không dùng
chẩn đoán này để sửa ngưỡng hoặc mở B2 confirmation.


**B3 follow-up sau B2 FAIL (18-10-2026 17:35 UTC): prospective, DEV đang chạy.**
Căn cứ B2 DEV-only: relational message đổi hành động và cải thiện chính trọng
số của full khi được bật, nhưng full thua mask-only huấn luyện riêng. B3
đăng ký *một* thay đổi biểu diễn: tổng duration của selected pair chia
`4*K` và tổng duration × `(1-restoration)` chia `4*K`; actor của cạnh ứng
viên nhận mỗi tổng nhân với duration/4, WAIT nhận hai tổng trực tiếp.
Physics/cost V4, teacher bank, PPO budget và ngưỡng không đổi. Protocol và
seed registry riêng đã push ở `0d23db8` trước mã và outcome; code `a99ab1c`
đã qua 7 targeted tests, smoke kỹ thuật timestamp `20261018T173224Z`
và verifier độc lập replay 2.400 physical ticks, 1.152 stage, zero invalid,
rồi commit/push trước DEV. Smoke có 1 bundle, 384 PPO decisions và 4
engineering TEST episodes nên **không** là bằng chứng khoa học B3. DEV mới
chạy dưới user service `codex-reservation-b3-dev-20261018T173438Z` với
3 bundle, N4/N8, 32 seed DEV/scale; panel confirmation 256 TEST seed/scale
giữ sealed cho đến khi DEV gate và kiểm toán độc lập cùng PASS. B1/B2 FAIL
và B1/B2 TEST sealed giữ nguyên.


**Chuẩn bị confirmation B3 (18-10-2026 17:38 UTC):** nhánh riêng
`codex/reservation-prefix-capacity-v3-confirmation`, commit `079001a` đã
push. Runner này yêu cầu DEV `COMPLETED`, gate nội bộ PASS, TEST chưa mở,
protocol/registry và mọi source module học/physics trùng snapshot DEV; thêm
điều kiện báo cáo audit DEV độc lập `DEV_GATE_PASS` khớp đúng manifest hash
và verifier hash. Bài kiểm tra source guard khớp snapshot DEV đang chạy;
9 targeted tests PASS. Verifier độc lập tăng phạm vi lên cả quỹ đạo full
và mask-only: smoke kỹ thuật 384+384 state, 2.304 decoder stage, 2.400
physical tick, zero invalid, PASS. Không thay learner, seed, budget hay ngưỡng.
Chỉ dùng runner nếu DEV B3 và audit độc lập cùng PASS; hiện TEST sealed.


**Gate audit độc lập cho confirmation A1/C1 (18-10-2026 17:42 UTC):**
A1 runner trên nhánh `codex/reservation-teacher-transfer-confirmation`,
commit `1b68a49` đã push, yêu cầu manifest DEV hoàn chỉnh, screen PASS,
TEST sealed và báo cáo audit độc lập PASS khớp đúng artifact path, manifest
hash, verifier hash. Protocol/registry và các module env/policy/distillation
khớp snapshot DEV live trong kiểm tra read-only; 8 targeted tests PASS.
C1 runner trên nhánh `codex/reservation-diverse-confirmation-v1`, commit
`d1a8418` đã push, thêm cùng điều kiện audit độc lập khớp exact DEV
artifact/manifest/verifier, và tiếp tục đóng băng rule FCFS/risk chọn bằng
DEV; 3 focused confirmation tests PASS. Hai thay đổi chỉ ở đường mở
confirmation, không sửa learner/physics/panel của DEV đang chạy. Chưa mở
A1/C1 TEST và chưa có verdict mới.


**B3 DEV verdict (18-10-2026 21:58 UTC): FAIL, TEST sealed.** Run
hoàn tất sạch sau 13.278,84 s (service exit 0, manifest `COMPLETED`,
audit nội bộ PASS), source commit `a99ab1c` đã push trước FULL. Verifier
độc lập tại `/home/bachng/.codex/worktrees/reservation-prefix-capacity-v3/ht-pdm-fjsp/artifacts/technician_prefix_capacity_v3_development_independent_audit_20261018T215834Z`
PASS, manifest SHA-256
`3a5c4e241dc6acb5e78741146bc07ae5dacfbb8b3aeb4601f8bd1e88c93cf403`,
kiểm 265 source seal, 480 output binding, 184.320 episode huấn luyện,
24 model, panel DEV và training tape ghép cặp. Gain full so với
mask-only học lại: pooled **+0,3646%**, dưới DEV gate **+0,5%**;
N4 **+1,1819%**, N8 **−0,4576%**. N8 có 2/3 replica full
cost cao hơn mask-only. Mọi health point guard, zero-invalid đạt; không
đổi verdict cost. B1/B2/B3 đều không đạt contrast prefix đã đăng ký;
không mở B3 confirmation hay dùng 256 TEST seed B3. Chẩn đoán tiếp
theo, nếu làm, chỉ trên DEV đã mở, phải giữ riêng khỏi claim khoa học.

**B3 chẩn đoán DEV cùng trọng số (18-10-2026 22:08 UTC):** kế hoạch
thăm dò đã khóa sau verdict B3, trước can thiệp; mã chẩn đoán `59ebdb6`
đã qua 3 targeted tests và smoke 12 episode, commit/push trước run.
Artifact `/home/bachng/.codex/worktrees/reservation-prefix-capacity-v3/ht-pdm-fjsp/artifacts/technician_prefix_capacity_v3_dev_diagnostic_20261018T220219Z`
`COMPLETED`, 192 episode (N4/N8 × 3 bundle × 32 seed DEV), 373,55 s,
`test_opened=false`. Audit độc lập chỉ trên artifact đã lưu kiểm toàn bộ
output hash, panel, replay full so với hàng DEV đã khóa và phép tổng hợp;
PASS tại `/home/bachng/ht-pdm-fjsp/artifacts/technician_prefix_capacity_v3_dev_diagnostic_independent_audit_20261018T230722Z` (report SHA-256 `14cd5b7f80327996db1914f4b78b8f090ce4130ddc64193026a0b9cb8f257f9b`), không tự rerun policy/physics. Tắt **riêng** hai input capacity trên
trọng số full đã fit đổi 0/4.608 tick ở N4 và 2/4.608 tick ở N8;
cost/machine N4 giữ 262,8047, N8 từ 265,5482 lên 265,5794. Tắt
cả capacity lẫn relational message trên cùng trọng số đổi 31/4.608
tick N4 và 470/4.608 tick N8; cost/machine lên 269,2005 và 270,4714.
Các con số này cho thấy relational message hiện đang chi phối nhiều quyết
định hơn hai capacity input mới, nhưng **không** là lợi ích so với mask-only
huấn luyện độc lập. B3 scientific DEV FAIL, N8 bất lợi, và B3 TEST sealed
giữ nguyên. Không chọn checkpoint/ngưỡng khác dựa trên chẩn đoán này.

**B4 follow-up sau B3 FAIL (18-10-2026 23:15 UTC): prospective, DEV đang chạy.**
Căn cứ từ chẩn đoán B3: capacity scalar mới gần như không đổi hành động;
relational prefix đổi nhiều hành động nhưng chưa thắng mask-only học riêng.
B4 thêm đúng một đường residual trên logit của mỗi cạnh ứng viên/WAIT:
tích vô hướng `query(edge)` với `key(selected-edge prefix)` dùng lại
trọng số graph hiện có, không thêm tham số hoặc thay physics/cost/budget.
Mask-only cùng mô hình khử selected-edge/capacity như trước; legal mask,
graph hiện tại và resource pool khớp. Protocol/ngưỡng/hardstop và registry
seed mới disjoint đã commit/push tại `8521427` **trước mã**; mã `0d94985`
đã qua 8 focused tests và smoke timestamp
`/home/bachng/ht-pdm-fjsp-prefix-residual-v4/artifacts/technician_prefix_residual_v4_smoke_20261018T231212Z`.
Smoke có 1 bundle và engineering TEST seed riêng, không là verdict B4.
Audit độc lập smoke PASS tại `/home/bachng/ht-pdm-fjsp-prefix-residual-v4/artifacts/technician_prefix_residual_v4_smoke_independent_verify_20261018T231403Z` (report SHA-256 `989892d6075ef5b916c67bbddf23b4118301901ca419021cb28c62268921f3d9`): 337 output bindings, 2.400 tick vật lý,
384 full-policy state/1.152 stage, 0 invalid; residual khác zero ở
570 stage và đổi 80 lựa chọn greedy trên trạng thái full (same-weight,
không phải contrast chính). Source code đã push trước DEV FULL. Run
`codex-reservation-b4-dev-20261018T231458Z` mở **chỉ** DEV N4/N8,
3 bundle × 32 seed DEV, artifact timestamp riêng; 256 TEST seed/scale
của B4 vẫn sealed. B1/B2/B3 FAIL và TEST sealed không đổi.

**Chuẩn bị confirmation B4 (18-10-2026 23:19 UTC):** nhánh riêng
`codex/reservation-prefix-residual-v4-confirmation`, commit `c1f8d5c`
đã push. Runner yêu cầu B4 DEV `COMPLETED`, internal audit PASS,
DEV screen PASS, TEST sealed, protocol/registry và các module học/physics
khớp snapshot DEV, cộng thêm independent DEV `DEV_GATE_PASS` gắn exact
manifest/verifier hash. Source guard đã PASS read-only trên snapshot
B4 DEV đang chạy; 10 focused tests PASS. Verifier TEST độc lập mở rộng
kiểm cả quỹ đạo full/mask-only, residual/logit, legal set và physics:
engineering smoke 384+384 state, 2.304 stage, 2.400 tick, zero invalid,
PASS. Đây là chuẩn bị đường mở TEST có điều kiện, không là kết quả khoa học
hay giấy phép mở TEST khi DEV chưa đạt.

**B4 DEV verdict (19-10-2026 03:32 UTC): FAIL, TEST sealed.** Run
`/home/bachng/ht-pdm-fjsp-prefix-residual-v4/artifacts/technician_prefix_residual_v4_development_20261018T231458Z`
hoàn tất sạch sau 15.364,86 s (service exit 0, manifest `COMPLETED`,
internal audit PASS), source commit `0d94985` đã push trước FULL.
Verifier độc lập tại
`/home/bachng/ht-pdm-fjsp-prefix-residual-v4-confirmation/artifacts/technician_prefix_residual_v4_development_independent_audit_20261019T033153Z`
PASS (report SHA-256
`bd26bee4950fb869ed9eedacc9cc2bbed6a36a93efb78506d94931090925a4bb`,
source manifest SHA-256
`bcb284ff6af94a0cdce7cd7172c8f42f32d35883fbfba4834d925e7901c24f9e`).
Audit kiểm 270 source seal, 485 output binding, 184.320 episode huấn
luyện, 24 model, 61.440 training tape ghép cặp, 3.072 episode DEV,
checkpoint/cost/health và `test_opened=false`.

Contrast full so với mask-only học riêng theo thước đo đã đăng ký:
pooled **−2,0386%** (gate DEV +0,5%), N4 **−0,1643%**,
N8 **−3,9177%**. N8 full availability 0,65823 so với mask-only
0,66897, chênh **−0,01074** vượt biên −0,01; pending/failure và
zero-invalid point guards còn lại đạt. Direct logit residual đã đổi
quyết định ở engineering smoke, nhưng không tạo matched-control gain
trên DEV mới. Giữ B4 verdict âm, không mở confirmation/TEST, không
thay ngưỡng hoặc chọn lại checkpoint. B1/B2/B3/B4 đều không xác nhận
claim B. Bước kế tiếp chỉ dùng train/DEV B4 đã mở để chẩn đoán residual
cùng trọng số; nếu có căn cứ cơ chế mới thì khóa một panel fresh khác
trước bất kỳ follow-up khoa học nào.

**B4 chẩn đoán DEV cùng trọng số (19-10-2026 03:43 UTC):**
kế hoạch thăm dò đã khóa tại `53f3727` sau verdict B4, trước can
thiệp; script/test `3688fae` đã qua 2 focused tests và smoke 12
episode, commit/push trước full. Smoke đầu dừng kỹ thuật vì kiểm tra
hiệu logit sai số làm tròn 1,9×10⁻⁶; đã sửa sang so phép cộng đúng thứ
tự forward, giữ artifact FAILED và smoke timestamp mới PASS.
Full DEV-only artifact
`/home/bachng/ht-pdm-fjsp-prefix-residual-v4/artifacts/technician_prefix_residual_v4_dev_diagnostic_20261019T033805Z`
`COMPLETED`, 192 episode, 308,60 s, `test_opened=false`.
Audit độc lập trên hash/panel/hàng DEV đã khóa và phép tổng hợp tại
`/home/bachng/ht-pdm-fjsp-prefix-residual-v4-confirmation/artifacts/technician_prefix_residual_v4_dev_diagnostic_independent_audit_20261019T034329Z`
PASS (report SHA-256
`c590037a5f5b9ff7c8230df54519af31dfdfc29aba540c296e23a131137b81ab`);
audit này không tự rerun policy/physics.

Tắt **chỉ** direct residual mà giữ encoder, relational prefix,
capacity, legal mask và mọi trọng số đã fit làm đổi 6/4.608 tick N4
và 1.490/4.608 tick N8 trên trạng thái full. Cost/machine full so
với residual-off là 263,5260 vs 264,3464 N4 và 272,7109 vs
273,4805 N8; full có lợi trên trung bình same-weight, nhưng replica
N8 seed 69000002 bất lợi (284,4414 vs 282,0625). Đây là bằng chứng
residual được dùng mạnh ở N8, không phải bằng chứng full vượt
mask-only huấn luyện riêng. B4 DEV_GATE_FAIL, N8 health fail và B4
TEST sealed giữ nguyên. B1–B4 cùng mô thức: tín hiệu prefix ảnh hưởng
policy đã fit nhưng không đạt contrast matched trained. Follow-up
tiếp theo, nếu làm, nên thử ổn định quá trình học từ cùng precursor
mask-only với budget và panel mới; không quét hệ số residual hay sửa
checkpoint/ngưỡng trên B4 DEV.

**B5 follow-up ổn định học (19-10-2026 03:58 UTC):** sau chẩn
đoán B4, khóa trước protocol/panel mới tại commit `703e5e2`, push trước
code/outcome. Giả thuyết duy nhất: hai arm dùng chung mask-only warm và
245.760 quyết định PPO đầu hoàn toàn đồng nhất; tại ranh giới rollout,
chỉ full bật relational prefix, capacity và direct residual cho 245.760
quyết định còn lại, Adam state giữ nguyên. Kiểm hash tham số, chuỗi
action, tiến trình và episode ở precursor; lệch là lỗi kỹ thuật. Cả hai
chọn checkpoint chỉ ở 245.760/368.640/491.520. Giữ nguyên cost/health
DEV gate B4, physics, budget 491.520, TEST sealed. Seed DEV 71000000–
71200000, confirmation mới 71400000–71700000, smoke riêng
71800000–72100000; audit raw IDs không trùng registry trước.

Code B5 `2995987` gồm verifier DEV/TEST độc lập, 8 focused tests
PASS; engineering smoke cuối `technician_prefix_activation_v5_smoke_20261019T041100Z`
`COMPLETED`, internal audit PASS, independent replay 2.400 physical tick
PASS. Tên thư mục smoke được gán trước giờ chạy; UTC thật nằm trong
manifest (03:56:26–03:57:07 UTC); không dùng smoke để quyết định khoa
học. Code đã push và đối chiếu remote exact SHA trước FULL. DEV FULL
artifact mới
`/home/bachng/ht-pdm-fjsp-prefix-activation-v5/artifacts/technician_prefix_activation_v5_development_20261019T035848Z`
đang RUNNING ở commit `2995987`, manifest `test_opened=false`. Chỉ khi
COMPLETED mới chạy verifier DEV; nếu fail, giữ B5 TEST sealed và không
đổi verdict B1–B4.

Runner B5 confirmation nằm ở checkout riêng
`/home/bachng/ht-pdm-fjsp-prefix-activation-v5-confirmation`, commit
`1625101` đã push. Nó khóa protocol/registry và mọi module học/physics
đúng snapshot DEV, chỉ ngoại lệ wrapper thêm gate; yêu cầu DEV COMPLETED,
internal audit PASS, DEV screen PASS, TEST sealed và verifier DEV độc lập
`DEV_GATE_PASS` gắn đúng manifest/verifier hash. 10 focused tests PASS;
read-only guard trên snapshot DEV đang chạy PASS và khi thiếu audit độc
lập nó từ chối mở TEST. Runner chưa được phép chạy confirmation.

**B5 DEV verdict (19-10-2026 08:17 UTC): FAIL, TEST sealed.** Run
`/home/bachng/ht-pdm-fjsp-prefix-activation-v5/artifacts/technician_prefix_activation_v5_development_20261019T035848Z`
`COMPLETED` exit 0 sau 15.439,52 s, internal audit PASS, source commit
`2995987` đã push trước FULL, 24 model và 1.920 DEV episode. Verifier
độc lập tại
`/home/bachng/ht-pdm-fjsp-prefix-activation-v5-confirmation/artifacts/technician_prefix_activation_v5_development_independent_audit_20261019T081719Z`
PASS nhưng verdict `DEV_GATE_FAIL` (report SHA-256
`0fa48ba2eadbfb3e382e4341fa10fe009756c2daef55afd825387b56efd27c8d`;
source manifest SHA-256
`66c242b7b65b0ec1f3d63fab654e67c60a59740cb210c0476f7037aea2ad5c60`;
verifier SHA-256
`9f450834f9b5c7210987241cd00418d7151f118371f096ec1294ffef93e8d710`).
Audit kiểm 275 source seal, 448 output binding, 31.200 precursor record
full/mask và branch checkpoint, checkpoint/cost/health; `test_opened=false`.

Matched selected-cost gain full so với mask-only là **−12,8136% pooled**,
N4 **−8,3221%**, N8 **−17,2770%**, thấp xa DEV gate +0,5%.
Availability full/mask N4 0,65272/0,66748, N8 0,60856/0,66838;
failures/machine N4 0,90104/0,12240, N8 0,97396/0,10807.
N4 pending/machine 0,05729/0; N8 0,01823/0. Zero-invalid guard
đạt, nhưng nhiều health guard khác fail. Không mở B5 confirmation/TEST,
không sửa checkpoint/ngưỡng hoặc tái diễn giải B1–B4.

Chẩn đoán mô tả **chỉ trên DEV đã mở**, không là kết quả mới: tại
checkpoint chuyển mode 245.760, cost/machine full/mask N4 297,7943/
263,0130 (gain −13,2242%), N8 383,2148/265,9141 (−44,1123%).
Sau thêm PPO đến 491.520, N4 285,7917/265,1198 (−7,7972%),
N8 312,7279/263,7747 (−18,5587%). Bởi precursor hash, action
sequence và record full/mask trước branch khớp, tổn thất lớn bắt đầu
ở việc bật prefix; quá trình học sau đó hồi phục một phần nhưng không
đạt matched contrast. Chưa định danh riêng tác dụng của relational
message, capacity và direct residual; cần chẩn đoán cơ chế DEV-only
trước khi khóa bất kỳ follow-up khoa học mới nào.

**B5 chẩn đoán cơ chế tại branch, DEV-only (19-10-2026 08:35 UTC):**
Kế hoạch can thiệp đã khóa và push tại `febac59` sau B5
`DEV_GATE_FAIL`, trước khi viết code. Script/tests đã qua 12 focused tests,
smoke 12 episode trên hai seed DEV đầu của mỗi scale, rồi commit/push
`3198cbc` trước full diagnosis. Artifact mới
`/home/bachng/ht-pdm-fjsp-prefix-activation-v5-confirmation/artifacts/technician_prefix_activation_v5_diagnostic_full_20261019T082808Z`
`COMPLETED` 192 episode, 348,36 s, manifest SHA-256
`5b6be57413edb3623b519fd20f6887d964f51fd4b7b67ade3bf8915bbea03d4c`.
Independent saved-artifact/panel/arithmetic audit tại
`/home/bachng/ht-pdm-fjsp-prefix-activation-v5-confirmation/artifacts/technician_prefix_activation_v5_diagnostic_independent_audit_20261019T083505Z`
PASS, report SHA-256
`dd0f9af10473d3cb3bc332182499cb581f7f68bf2d1bd19496824f7104d551a8`.
Checkpoint full/mask 245.760 từng bundle có parameter digest bằng nhau;
full và mask replay khớp đúng các hàng B5 DEV đã lưu, physical tape
khớp, zero invalid. Can thiệp cùng trọng số chỉ triệt direct logit
residual sau graph encoder, giữ nguyên legal mask, resource, stats,
relational prefix, capacity và base logits trong mọi trace.

Cost/machine trung bình tại branch N4: full **297,7943**, residual-off
**263,0130**, mask **263,0130**; N8: full **383,2148**,
residual-off **266,4740**, mask **265,9141**. Residual-off đổi
1.123/4.608 full-state action N4 và 3.613/4.608 N8, so với full;
logit residual khác zero ở 3.060/9.216 stage N4 và 12.629/18.432
stage N8. Như vậy phần lớn tổn thất *tức thời* khi bật B5 là do
direct residual; relational/capacity một mình tại branch với trọng số
chưa fine-tune gần mask-only. Đây là phép can thiệp thăm dò trên DEV,
không chứng minh hiệu quả sau đào tạo hoặc tác dụng nhân quả của các
thành phần khi đã học. B5 `DEV_GATE_FAIL` và TEST sealed giữ nguyên.
Bằng chứng này đủ để thiết kế **một** follow-up prospectively: bỏ
đường direct residual, giữ relational/capacity với shared mask-only
precursor và budget bằng mask-only, dùng train/DEV/conditional TEST
seed mới, không quét coefficient hoặc chọn lại B5 checkpoint.

**B6 thiết kế prospective, chưa chạy (19-10-2026 08:38 UTC):**
`docs/technician_prefix_residual_free_v6_plan.md` khóa đúng một cơ chế
follow-up: shared mask-only precursor, bật relational/capacity prefix
sau 245.760 quyết định nhưng triệt direct logit residual trong full arm;
control mask-only được huấn luyện riêng với cùng ngân sách. Không thay
đổi DEV/confirmation gates B5, không dựa trên B5 TEST. Registry
`configs/technician_prefix_residual_free_v6_seed_registry.json` dùng
370 raw seed mới (train/DEV/conditional TEST/smoke), kiểm tra không
trùng các registry hiện hữu. Đây là thiết kế từ DEV diagnosis; tại thời điểm khóa chưa
có efficacy result hoặc B6 run.

**C1 DEV gate và confirmation (19-10-2026 09:02 UTC):** C1 run
`/home/bachng/ht-pdm-fjsp-diverse-v1/artifacts/technician_diverse_maintenance_v1_development_20261018T134420Z`
COMPLETED exit 0, internal audit PASS, `test_opened=false`, source
`396ecb5` đã push, 48 model, 7.680 DEV episode, 68.335,44 s wall.
Verifier độc lập tại
`/home/bachng/ht-pdm-fjsp-diverse-v1/artifacts/technician_diverse_maintenance_v1_development_independent_audit_20261019T090024Z`
PASS, report SHA-256
`de048191e7e12acac592c665350823867bed92b67722a58bce90c523caed0eba`,
source manifest SHA-256
`0c78bc97e855f298142dd6958b4cae28b14a1c8a3f37351e2ffdcc6daccc146c`.
Kiểm 265 source seal, 691 output binding, 7.936 hàng DEV/reference,
491.520 training episode và 128 physical tape DEV ghép cặp. DEV screen
PASS: pooled full/mask gain **12,9747%**, long_service **6,9969%**,
skill_bottleneck **20,2024%**, gain so với heuristic risk đã chọn trên
DEV **9,4295%**; tất cả family health point guard PASS. Đây chưa là
claim TEST.

Confirmation C1 mở bằng runner `d1a8418` đã push, 14 focused tests
PASS, preflight bind đúng DEV manifest/report/verifier hash. Unit
`codex-reservation-c1-confirmation-20261019T090238Z`, artifact mới
`/home/bachng/ht-pdm-fjsp-diverse-confirmation-v1/artifacts/technician_diverse_maintenance_v1_confirmation_20261019T090238Z`
RUNNING; 5 training bundle và TEST mới được dùng theo protocol khi đến
đúng stage. Tại lúc ghi, `test_opened=false`, chưa có confirmation
verdict. Sau COMPLETED phải chạy verifier TEST độc lập trước kết luận C.

**B6 triển khai theo plan đã khóa (19-10-2026 09:09 UTC):** bản sao plan/
registry đúng hash được push trước code tại `0b8f936`; residual-free
policy, DEV auditor và TEST verifier ở `cfa3937` đã push sau 16 focused
regression tests, engineering smoke 384 quyết định/1 bundle và
independent physical/prefix replay PASS (2.400 tick, zero invalid,
zero direct-residual stage, 442 nonzero prefix stage). B6 DEV unit
`codex-reservation-b6-dev-20261019T090913Z`, artifact
`/home/bachng/ht-pdm-fjsp-prefix-residual-free-v6/artifacts/technician_prefix_residual_free_v6_development_20261019T090913Z`
RUNNING trên panel DEV mới; `test_opened=false`. Confirmation runner
riêng `7634d8d` đã push, 19 tests PASS, engineering smoke và replay
độc lập PASS; guard từ chối DEV RUNNING và yêu cầu exact independent
`DEV_GATE_PASS` bound bằng hash. B6 chưa có khoa học DEV verdict;
TEST khoa học vẫn sealed.

**B6 DEV verdict (19-10-2026 13:48 UTC):** run trên hoàn tất exit 0,
internal audit PASS, 24 model, 1.920 DEV episode, `test_opened=false`.
Verifier độc lập tại
`/home/bachng/ht-pdm-fjsp-prefix-residual-free-v6/artifacts/technician_prefix_residual_free_v6_development_independent_audit_20261019T134836Z`
PASS cho tính đúng và tái tính đúng verdict **`DEV_GATE_FAIL`**; report
SHA-256 `71cad2c5825c6029396181f2789d03265067bdcc882722f0ac8b83253aa28a26`,
source manifest SHA-256
`48b734fb26efe36e72bc9ad49a67f2dba1192376a628ab08d4374ee4b4425019`.
Selected pooled full/mask cost gain chỉ **+0,0933%**, dưới ngưỡng DEV
+0,5%; N4 **+0,2566%**, N8 **−0,0729%**. Các point health guard và
zero-invalid đều đạt. Verifier kiểm 280 source seal, 453 output binding,
24 training audit, 31.200 precursor record và shared branch checkpoint.
Việc bỏ direct residual loại phần hại tức thời của B5 nhưng không tạo
được lợi ích prefix đủ mức khi huấn luyện lại với mask-only control.
Giữ B6 scientific TEST sealed; confirmation runner không được mở.
B1–B6 đều không đạt DEV gate, không suy ra hiệu quả prefix độc lập.
Chẩn đoán tiếp theo, nếu có, chỉ dùng B6 train/DEV đã mở và phải khóa
giả thuyết trước một follow-up mới với seed/panel khác; không quét hậu
nghiệm các checkpoint hay mở lại TEST B1–B6.

**Giới hạn staffing/generalization:** V4 chỉ dùng `K=N/2`; C1 đang xác
nhận hai họ non-V4, trong đó `skill_bottleneck` dùng `K/N=3/8=6/16`
còn `long_service` vẫn dùng `K/N=1/2`. Đây không phải phép chứng minh
khái quát trên mọi tỷ lệ technician/máy. Tỷ lệ phù hợp còn phụ thuộc
arrival load, service duration và skill coverage. Sau verdict C1, nếu
muốn claim độ rộng staffing, phải đăng ký panel seed mới thay đổi độc
lập `K/N`, topology kỹ năng và mức tải; báo zero-shot, fine-tuning và
train mới riêng, không diễn giải fine-tuning là zero-shot.


**Diễn giải B/C khóa khi C1 TEST còn sealed (19-10-2026 13:53 UTC):**
C1 protocol được khóa trước mọi C1 outcome và cả khi B2 DEV còn chạy.
Nó gồm full relational selected-edge prefix, mask-only huấn luyện riêng
với cùng N4 teacher/PPO budget/legal set/order, và no-reservation/no-prefix
cùng teacher/budget. C1 đã đăng ký contrast full/mask trên bốn cấu hình
non-V4 với point gain >=1%, cận dưới CI một phía sau Bonferroni >0,
family/health guards. Do đó, nếu confirmation C1 đạt và verifier độc lập
PASS, cùng panel này có thể hỗ trợ claim B **chỉ trong hai họ non-V4 đã
đăng ký**, đồng thời là một phần claim C. Điều này không đảo verdict âm
B1–B6 trên balanced V4, không chứng minh prefix hữu ích ở mọi cấu hình,
và không cho phép đổi protocol sau khi nhìn TEST. Nếu C1 không đạt, cả
B lẫn C liên quan vẫn chưa được xác nhận. C1 TEST chưa mở tại thời điểm
ghi chú này.
