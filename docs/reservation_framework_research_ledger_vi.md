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
