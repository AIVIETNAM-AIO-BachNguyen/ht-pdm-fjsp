# Xác nhận cấu hình technician 2–2 trên Mac

Material Passport: dữ liệu tổng hợp từ simulator; một cấu hình cố định do người
nghiên cứu yêu cầu; không có training RL; raw seed mới, paired theo method;
kiểm tra scalar physics và indexed bootstrap tách khỏi runner. Bộ kiểm tra được
kế thừa từ implementation viết độc lập trước đó và chỉnh theo protocol mới
trong cùng ngữ cảnh; không phải kiểm tra mù hoặc cross-model.

**Kết luận: NOT_CONFIRMED, đạt 17/18 guard.** Joint có bằng chứng tốt trong lượt
xác nhận này, nhưng allocation-only chưa chứng minh availability không giảm
quá1 điểm phần trăm. Không đổi ngưỡng, chọn lại baseline hoặc mở seed khác.

Hai technician đều làm được bốn máy. Technician A nhanh trên máy1–2, B nhanh
trên máy3–4. Làm nhanh1tick/phục hồi50%; làm chậm3tick/phục hồi100%. Giữ nguyên
48policyticks +12common deadline tail, costs/failure/urgency và rollout128CRN.
Baseline isolated_endpoint đã chọn trên16developmentseed của protocol trước,
không chọn lại trên confirmation. Cấu hình này từng fail screening24seed;
lượt này là standalone author-requested confirmation, không đổi lịch sử.

## Kết quả dễ đọc

- Joint giảm **11,6354%** chi phí so với baseline cố định; CI99,722%
  **[9,4393%;13,7913%]**. So với deadline giảm13,2639%; so với h6 giảm15,1420%.
- Joint availability **66,6951%**, baseline65,3178%: tăng **1,3774pp**;
  CI99,722%[0,6099;2,2511]pp; pending tại H=0. Tất cả guard joint đạt.
- Allocation-only giảm **6,3276%** chi phí; CI99,722%[4,5194%;8,2210%].
  Availability64,6810%, giảm **0,6368pp**; CI99,722%[-1,3778;0,0920]pp.
  Cận dưới−1,3778pp < ngưỡng−1pp nên **FAIL noninferiority**. Đây là chưa
  chứng minh được giới hạn giảm availability; không phải bằng chứng chắc chắn
  rằng mức giảm thật vượt1pp.
- Local root assay giữ nguyên state/tick/subset: allocation đã chọn trên một
  selection bank, đánh giá trên holdout riêng, tốt hơn isolated endpoint DP
  **6,6802cost/common landmark**; CI99,722%[4,5717;8,9655]. Bao gồm WAIT và
  các landmark không có lựa chọn; bootstrap theo256rawseed, không theo1792row.

Có thể dùng2–2 làm cấu hình chính để nghiên cứu joint maintenance và cho thấy
tradeoff phân công technician rõ ràng. Tuy nhiên, theo bộ tiêu chí đã khóa,
chưa được viết rằng2–2 đã xác nhận đầy đủ cho allocation-only giữ health.
Không có so sánh thống kê trực tiếp2–2 với3–1 vì hai lượt dùng seed khác nhau.
Không suy ra chất lượng PPO/MARL từ kết quả rollout này.

## Evidence và tái lập

Code commit `f56f5bf4fc17e6ddab4e69af0b4f5205a7d8e08d` đã push trước full, branch
`codex/maintenance-balanced-confirmation-mac-v1`. Targeted regression24PASS
trong19,04s; CLI smoke và smoke independent verifierPASS. FullCPU Mac hoàn tất
282.303s,1280episodes/76800physicalticks/1792landmarks,
3047rootrows,366078976forecasttransitions; zero invalid assignments.
Independent auditPASS:62artifacthashes,51committedsourceblobs, scalar states,
physical shocks, costs, frozen choices và18indexed-bootstrap CIs.

Artifact: `/Users/bachng/Coding/Reinforcement Learning/marl/artifacts/maintenance_balanced_confirmation_mac_v1_full_cpu_20261012T084759Z`

Verifier command:

```bash
cd /private/tmp/marl-maintenance-assignment-confidence-v1 && .venv/bin/python scripts/verify_maintenance_balanced_confirmation.py '/Users/bachng/Coding/Reinforcement Learning/marl/artifacts/maintenance_balanced_confirmation_mac_v1_full_cpu_20261012T084759Z' /private/tmp/marl-maintenance-assignment-confidence-v1
```

## Means

| Controller | H+L cost | Availability | Pending at H |
|---|---:|---:|---:|
| deadline_matching | 656.035 | 65.420% | 0.0547 |
| isolated_h6 | 670.555 | 59.473% | 0.0039 |
| isolated_endpoint | 643.945 | 65.318% | 0.0000 |
| assignment_rollout | 603.199 | 64.681% | 0.0312 |
| joint_rollout | 569.020 | 66.695% | 0.0000 |

## Eighteen preregistered comparisons

Cost gain and availability values below are fractions; local gains are cost
per common landmark. EachCI confidence1−.05/18, bootstrap20,000seedclusters.
Approximate simultaneous inference applies to this protocol, not all prior
searches. All18guards required; no subset promotion.

| Metric | Estimate | CI99.722% | Gate |
|---|---:|---|---|
| local_baseline | 6.680219 | [4.571653, 8.965504] | PASS |
| local_quality | 6.680219 | [4.571653, 8.965504] | PASS |
| local_endpoint_dp | 6.680219 | [4.571653, 8.965504] | PASS |
| assignment_cost_gain | 0.063276 | [0.045194, 0.082210] | PASS |
| joint_cost_vs_deadline_matching | 0.132639 | [0.109629, 0.155498] | PASS |
| joint_cost_vs_isolated_h6 | 0.151420 | [0.129099, 0.172228] | PASS |
| joint_cost_vs_isolated_endpoint | 0.116354 | [0.094393, 0.137913] | PASS |
| assignment_availability_difference | -0.006368 | [-0.013778, 0.000920] | FAIL |
| assignment_pending_difference | 0.031250 | [0.000000, 0.078125] | PASS |
| assignment_absolute_availability | 0.646810 | [0.637056, 0.655217] | PASS |
| assignment_absolute_pending | 0.031250 | [0.000000, 0.078125] | PASS |
| assignment_interior_gain | 0.054185 | [0.037676, 0.071641] | PASS |
| joint_availability_difference | 0.013774 | [0.006099, 0.022511] | PASS |
| joint_pending_difference | 0.000000 | [0.000000, 0.000000] | PASS |
| joint_absolute_availability | 0.666951 | [0.665527, 0.668258] | PASS |
| joint_absolute_pending | 0.000000 | [0.000000, 0.000000] | PASS |
| joint_interior_gain | 0.110492 | [0.089690, 0.131319] | PASS |
| assignment_choice_fraction | 0.700335 | [0.650112, 0.750000] | PASS |

## Eleven interpretation checks

1.No test tuning. 2.No optional stopping. 3.No silent fallback.
4.No landmark pseudoreplication. 5.Separate selection/holdout banks.
6.Zero-choice landmarks retained. 7.Tail included. 8.Health guard reported.
9.Only current root isolated; future decisions may change. 10.No optimality or
learning claim. 11.No unlimited-search coverage claim.
