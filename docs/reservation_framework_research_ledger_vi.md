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

## Protocol đang chạy, chưa có verdict khoa học

| Nhánh | Giả thuyết đã khóa / source commit của run | Artifact DEV | Trạng thái tính đến 18-10-2026 12:35 UTC |
|---|---|---|---|
| A1 teacher→student | Teacher N4 + distillation giúp graph reservation trên N8/N16/N32 so với scratch cùng PPO budget; `82c1989` | `/home/bachng/ht-pdm-fjsp-teacher-transfer-v1/artifacts/technician_teacher_transfer_v1_development_20261018T120344Z` | RUNNING; TEST sealed. Verifier độc lập đã PASS trên smoke, chờ DEV. |
| B2 relational prefix | Candidate-conditioned attention trên các selected edge giúp hơn mask-only học lại, sau B1 FAIL; `edbab65` | `/home/bachng/ht-pdm-fjsp-relational-v2/artifacts/technician_relational_prefix_v2_development_20261018T121324Z` | RUNNING; TEST sealed. Verifier được kiểm trên B1 artifact âm. |
| C1 đa cấu hình | N8/N16 skill/K khác V4 và service 2–4 tick với request bền vững, đối chứng mask-only/scratch/FCFS/risk; `27e896f` | `/home/bachng/ht-pdm-fjsp-diverse-v1/artifacts/technician_diverse_maintenance_v1_development_20261018T122731Z` | RUNNING; TEST sealed. Verifier độc lập đã PASS trên smoke. |

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
