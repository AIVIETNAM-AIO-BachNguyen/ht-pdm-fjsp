# Typed-Gibbs: chẩn đoán tối ưu hóa trên TRAIN/CFDEV

**Không mở thử nghiệm tăng gấp đôi budget.** Readout đã khóa không đạt; không sửa điều kiện sau số liệu. Kết quả xác nhận4/12 và mọi test cũ giữ nguyên.

Đã đọc75checkpoint nodes của3core×5replicas×5steps, cùng240trạng thái ghi sẵn từ5candidate trajectories trên đúng1CFDEVphysicalseed. Mỗi model dùng chung pool cố định:18.000policy-state evaluations;0rollout/0physicalstates mới/0trainingupdates. Không decode TEST performance trong chẩn đoán; chỉ dùng verdict âm và hashes làm entry gate.

Feasible matching probabilities tính từ NumPyMLP và reference21matching. Entropy chuẩn hóa theo log(feasible support); loại trạng thái chỉ WAIT khỏi entropy/collapse fractions. TV đo giữa368.640 và491.520 trên cùng pool, không suy ra cost từ trạng thái counterfactual.

| Candidate seed | Late CFDEV cost gain | Mean TV trên fixed pool | Fraction mode ≥.99 | Normalized entropy cuối | Unique MAP changes |
|---|---:|---:|---:|---:|---:|
| 380300000 | -0.725885% | 0.00008608 | 1.000000 | 0.00101689 | 0/240 |
| 380300001 | 0.399912% | 0.00184607 | 1.000000 | 0.00855510 | 0/240 |
| 380300002 | 0.561525% | 0.00071548 | 0.979167 | 0.00481151 | 0/240 |
| 380300003 | 0.434520% | 0.00594136 | 0.916667 | 0.02094473 | 0/240 |
| 380300004 | 2.641150% | 0.00027707 | 1.000000 | 0.00259824 | 0/240 |

Pooled lateCFDEVgain=0.678345% (cần≥.5%); positive replicas=4/5 (cần≥4); mean lateTV=0.00177321 (cần≥.01). Hai điều kiện đầu đạt, điều kiện chuyển động policy không đạt. `permit_designing_one_2x_budget_followup=false`.

Không có unique MAP change trên240fixed states ở đoạn cuối của bất kỳ trong15core models. Chỉ số TRAIN vẫn có actor parameter movement, nhưng đó không đồng nghĩa phân phối quyết định tiếp tục thay đổi. Readout possible-collapse flags là1/5Gibbs,4/5myopic,3/5reverse theo định nghĩa conjunctive đã khóa; không đồng nghĩa các model còn lại không có vấn đề.

Pool chỉ có1exogenousCFDEVseed, nên không đại diện toàn bộ64CFDEVseeds. Cost trên64seeds vẫn có thể cải thiện nhờ hành vi trên trạng thái không nằm trong pool. Vì vậy kết quả không chứng minh hội tụ toàn cục, nguyên nhân gradient, hay việc tăng budget chắc chắn vô ích. Nó chỉ không cung cấp entry gate đã yêu cầu cho phép thử2x; không tự động đổi gate hoặc chọn checkpoint tốt hơn.

Nguồn entropy hiện tại là exact physical matching entropy chiaN=4; cả ba core giống nhau, nên original causal contrast vẫn hợp lệ. Joint PPO likelihood/advantage có scope whole matching, còn PPO catalogue benchmark dùng whole matching entropy. Một phép thử chuẩn hóa entropy khác có thể được đăng ký riêng; đó là tuning với giả thuyết và factorial controls, không phải lỗi toán học đã được chứng minh, không phải đóng góp objective mới và không được gọi là kết quả thành công trước dữ liệu mới. Phép diagnostic entropy alias-opposition ở broker cũ đã âm; câu hỏi normalization magnitude này khác, không đảo verdict cũ.

Unit checks18PASS3,45s (4new+14reference); code60feac5 đã push trước run. Offline COMPLETED/auditPASS0,58s. Independently recomputed saved-array TV=0,00177321205779 và mọi mass1; source/protocol/input hashes giữ nguyên. CLI import launcher ban đầu sai sys.path trước khi tạo artifact; đã dùng đúng direct-script CLI, không rerun hay ghi đè experiment nào.

Artifact: `/home/bachng/ht-pdm-fjsp/artifacts/technician_typed_gibbs_optimization_audit_v1_20261017T131520448447Z`.

Goal ACTIVE/unachieved, N>4 đóng.
