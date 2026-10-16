# Fixed-broker aliasing diagnostic v1 — kết quả và giới hạn

**Diagnostic qua ngưỡng engineering đã khóa ở cả3replicas. Đây chưa phải kết quả giảm cost.** FULL gấp đôi budget đã thất bại11/13 và không được sửa verdict bằng diagnostic này. Scientific test và larger N/K vẫn đóng. Hướng mới cần một protocol học từ đầu riêng, kiểm tra chi phí/sức khỏe và ablation công bằng.

## Câu hỏi cụ thể

Khi nhiều bộ đề xuất cùng dẫn tới một phân công thực thi, gradient trên proposal có thể chứa nhiễu từ các đề xuất bị broker bỏ qua. Proposal PPO vẫn là policy-gradient estimator hợp lệ, không phải lỗi implementation. Câu hỏi ở đây là phần nhiễu đó có đáng kể trong actor đã học hay chỉ là một khả năng đại số.

Nguồn là ba `independent_raw_ppo` replicas của FULL budget-study frozen `4ade42381681e073efc8a2c7dc9efb43640a3809`. Không học, không rollout, không teacher, không đổi checkpoint selection, không mở seed mới. Tại mỗi replica dùng32DEV seeds, ticks0,3,...45 từ selected greedy trajectories:512contexts/replica. Cùng panel đó được dùng tại cả9checkpoints0..983040: tổng1536states,13824checkpoint-context evaluations,27frozen checkpoints.

Actor raw-state MLP có11141parameters dùng chung cho hai technician. Tính Jacobian của10logits theo TOÀN BỘ actor parameters ở float64 tại stored weights và inputs; Gram10x10. So sánh Fisher trace của categorical proposal scores với scores sau khi marginalize qua fixed broker. Closed-form mass/score được kiểm lại bằng tối đa25proposal vectors trong diagnostic N4 này, không đưa enumeration vào policy hoặc likelihood.

Đây là Fisher của **unit-weight actor score trên fixed selected greedy DEV states**. Panel này không phải on-policy state sample cho mọi checkpoint. Không được gọi39–76% dưới đây là mức giảm variance của return-gradient/PPO optimizer hay mức cải thiện cost.

## Primary checkpoint122880, khóa trước

| Train seed | Proposal Fisher mean | Physical Fisher mean | Fraction bỏ được | Entropy/technician (nats) | Conflict probability |
|---|---:|---:|---:|---:|---:|
| 280000000 | 22,405753 | 8,275901 | 63,0635% | 0,121623 | 0,187056 |
| 280000001 | 23,353944 | 14,192855 | 39,2272% | 0,112917 | 0,125715 |
| 280000002 | 19,102465 | 4,589987 | 75,9718% | 0,131131 | 0,165067 |

Fraction là ratio of SUMS trên512contexts, không phải mean của các ratio gần zero. Mỗi replica phải đạt fraction>=0,10, entropy>=0,10 và conflict probability>=0,05; cả9 điều kiện PASS. Tất cả27checkpoint summaries được giữ trong `summary.json`, không đổi primary sang checkpoint thuận lợi hơn. Ở checkpoint983040 fraction lần lượt41,39%/78,14%/53,55%, nhưng entropy giảm còn0,0112/0,0330/0,0132nats; các giá trị này không thay primary gate.

## Kiểm chứng và lỗi kỹ thuật đã giữ lại

Diagnostic hoàn tất38,90giây, audit PASS; có source/verification/checkpoint/state/observation/code hashes,13824CSV rows và27summaries. 21 deterministic unit tests PASS trước commit/push/execute. Chúng bao gồm mass, score, normalization, variance, exact support, full shared-actor Gram và ratio-of-sums.

Kiểm chứng độc lập sau đó PASS581bindings, toàn bộ13824-row grid và27arithmetic summaries,1536state/observation bindings,27checkpoint hashes. Nó còn kiểm27contexts thực bằng cách autodiff log mass của từng proposal trực tiếp theo toàn bộ11141parameters, rồi group proposal parameter scores theo matching. Phép kiểm này không gọi Gram/moment/closed-score helper của diagnostic. Context cho mỗi replica là earliest(seed,tick) có cả hai technician rảnh và ít nhất hai máy urgent; dùng cùng context đó cho cả9checkpoints. Các trace khớp trong tolerance2e-10 absolute+relative. Scope là27direct contexts, không phải tính lại độc lập toàn bộ13824Jacobians.

Lần verification đầu đã hoàn thành numerical checks nhưng lỗi ghi JSON do `numpy.bool_`; artifact FAILED được giữ nguyên. Fix chỉ chuyển engineering-check booleans sang kiểu Python để serialize; thêm regression test, tổng23tests PASS1,69s, commit/push rồi chạy lại vào timestamp mới. Không sửa diagnostic, FULL verdict, numerical thresholds hoặc data. Verification thành công sau fix mất1,52giây.

## Liên hệ thuật toán, chưa phải novelty/performance claim

Công thức xác suất matching sau winner-only broker nằm trong [derivation note](technician_broker_marginal_algebra_note.md). Nó factorizes một observed matching mà không liệt kê proposal/matching: winner phải request đúng máy; idle technician có thể WAIT hoặc request một máy đã có winner ưu tiên cao hơn. Conditional independence của proposals cho product of sums.

Generic marginal policy gradients và variance argument đã có trong [CAPG](https://proceedings.mlr.press/v80/fujita18a.html) và [Marginal Policy Gradients](https://arxiv.org/html/1806.05134#S4). Không tuyên bố phát minh nguyên lý này. Phần có thể phát triển là một likelihood có công thức tractable cho resource-arbitration, ghép vào PPO chuẩn và kiểm chứng bằng thực nghiệm allocation. Formula không áp dụng nếu broker rank phụ thuộc learned bids, có loser retry, global matching solver hoặc policy dùng prefix phụ thuộc proposal trước.

Lower variance của unclipped score không bảo đảm multi-epoch clipped PPO variance/cost tốt hơn. Exact physical entropy không bằng tổng proposal categorical entropy. Một experiment mới phải khóa rõ treatment entropy, giữ nó khớp giữa arms và không đánh tráo proposal entropy thành physical entropy. Greedy proposals cũng không phải global mode của induced physical distribution.

## Quyết định

Có đủ evidence để **thiết kế một phép thử có giới hạn**, chưa đủ để kết luận phương pháp thắng. Phép thử cần physical-ratio PPO và proposal-ratio PPO có cùng actor/critic/info/initial tensors/private updates/budget/broker/execution; thêm rank controls và cả hai PPO/MAPPO baselines. No teacher/warm-start. Chỉ khác likelihood learning trong contrast chính; không đổi encoder/LR để kéo kết quả. Original negative budget result giữ nguyên. CF và larger N/K chỉ mở khi gates mới đã khóa được đáp ứng và kiểm chứng độc lập.

## Artifacts và commits

- Conditional plan `bc2b372`; diagnostic frozen `e00fb59aee6982ea5dc249cdf3d9b06e8f382c9a`; algebra plan `97459ff`, algebra tests `2c1b6d0`.
- Diagnostic: `artifacts/technician_broker_aliasing_diagnostic_v1_20261016T140144414038620Z`.
- Verification procedure `63d4f92` (đăng ký sau khi nhìn diagnostic, chỉ scope verification); verifier `48a5f27`; serialization fix `b17b798`.
- Preserved FAILED verification: `artifacts/technician_broker_aliasing_verification_20261016T140909160595885Z`.
- PASS verification: `artifacts/technician_broker_aliasing_verification_20261016T141123400054613Z`.
- Branch: `codex/broker-marginal-algebra`.
- Source FULL: `artifacts/technician_raw_reservation_budget_v1_development_20261016T133319602371704Z`.
- Source independent verification: `artifacts/technician_raw_reservation_budget_v1_verification_20261016T140106818607718Z`.
