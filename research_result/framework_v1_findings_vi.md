# Kết quả framework điều phối bảo trì bằng RL, vòng v1

**Trạng thái:** FULL hoàn tất và audit độc lập PASS. Đây là benchmark tổng hợp
để phát triển phương pháp, không phải dữ liệu nhà máy hay chứng minh tối ưu.
Source commit `2fc6840a63fed765ec1503006765ec55e8dfc9d9`; run
`research_result/framework_v1_full_20261019T153014Z/`; audit
`research_result/framework_v1_audit_20261019T154240Z/`; diagnostic mở TEST
`research_result/framework_v1_diagnostic_20261019T154349Z/`.

## Bài toán và phương pháp

Bốn máy dùng chung hai technician, bảo trì kéo dài nhiều tick, request PM tồn
đọng và có thể chuyển CM; chỉ hoàn tất dịch vụ mới phục hồi. RL tập trung nhận
trạng thái hiện tại, chọn một matching hợp lệ gồm tập request cần phục vụ,
technician tương ứng hoặc WAIT ở từng tick. Mask thực thi ràng buộc; không coi
zero invalid assignment là năng lực policy tự học. Requester age≥3 cố định;
RL không học thời điểm gửi request. Reward là âm chi phí kinh tế H48+tail16;
không có penalty chờ nhân tạo. Bốn case cùng chi phí, objective và physical
seed; rollout dùng scenario forecast độc lập, không thấy shock thực tương lai.

Ba PPO training seeds, 199.680 bước mỗi seed; 12 checkpoint được chọn bằng 16
DEV seed chỉ trên anchor. Chọn seed `62000002` ở 199.680 bước, DEV mean 812,06.
Sau lựa chọn mới mở 64 TEST seed/case. Tổng 599.040 bước train, 192 episode DEV,
1.024 episode TEST; runtime campaign 726,8 giây. Tests liên quan: 48 pass;
Ubuntu smoke mới nhất `framework_v1_smoke_20261019T152916Z` hoàn tất 1.024
bước, 2 DEV và 32 eval episode. Checkpoint và raw episode không nằm trong Git.

Audit độc lập kiểm tra hash source/artifact, seed split, checkpoint selection,
đủ paired grid, physical seed, reward/cost/tail, request conservation và mọi
mean/paired difference trong summary. PASS: 12 checkpoint, 192 DEV, 1.024 TEST.
Không có assignment sai. Không kiểm chứng lại từng physical transition của
run v1 bằng một scalar engine thứ hai; kernel dùng ở đây đã có full diagnostic
trước đó với 221.184 transition replay độc lập, và unit tests bao phủ
completion, lifecycle, terminal, future tape và legal matching.

## TEST chính thức

Mean cost H+tail (64 seed/case); cost thấp hơn tốt hơn. Số trong ngoặc là paired
PPO trừ risk, CI bootstrap 95% đã khóa. `rollout` là bounded 6-tick/16-scenario
reference, không phải oracle tối ưu.

| Case | FCFS | Risk | Rollout | PPO | PPO − Risk [CI95%] |
|---|---:|---:|---:|---:|---:|
| Reciprocal scale 2, anchor | 914,39 | 839,91 | 695,62 | 817,45 | −22,45 [−60,94; 12,92] |
| Interchangeable scale 1 | 784,56 | 783,97 | 689,38 | 815,50 | +31,53 [14,01; 49,29] |
| Reciprocal scale 3 | 1.169,22 | 1.014,48 | 1.058,19 | 973,58 | −40,91 [−78,35; −4,43] |
| Specialist/generalist scale 2 | 1.013,27 | 1.010,17 | 981,53 | 1.060,30 | +50,12 [36,81; 63,88] |

H1 đã đăng ký, PPO tốt hơn risk trên anchor, **chưa được xác nhận** vì CI
chứa 0. PPO kém rollout trên anchor 121,83 cost/episode [106,87; 136,67].
Các case phụ là kiểm tra chuyển setting đã khóa, không phải panel được chọn để
PPO thắng. Reciprocal scale 3 là kết quả dương theo paired interval mô tả,
nhưng bốn contrast chưa hiệu chỉnh familywise. Giữ nguyên hai case PPO thua.

Thời gian chọn action trung bình mỗi episode ở anchor: PPO 0,0273 giây, risk
0,0014 giây, rollout 0,7001 giây. Rollout dùng trung bình 11.082 simulated
transitions/episode; phép so sánh cost không phải so sánh cùng compute.

## Diagnostic sau khi mở TEST (exploratory)

Trace của đúng checkpoint trên 12.288 quyết định của TEST đã mở cho thấy:

- Specialist/generalist: PPO WAIT dù có action phục vụ hợp lệ ở 246/3.072
  tick (8,01%); trên anchor chỉ 30/3.072 (0,98%). CM queued exposure là
  21,77 so với risk 6,19 machine-tick/episode; H availability 46,38% so với
  risk 51,78%. Đây là dấu hiệu thiếu thích nghi với hard skill exclusion.
- Interchangeable: PPO có H failures 7,55 so với risk 5,44; cost tăng 31,53.
  Trên anchor PPO có availability cao hơn risk (60,30% so với 57,34%) nhưng
  H failures 7,77 so với 6,64. Chất lượng không đồng nhất giữa availability
  và failure.
- Reciprocal scale 3: policy trùng risk ở 97,62% tick, nhưng CM queued
  exposure chỉ 1,25 so với risk 9,03 và cost thấp hơn. Một số quyết định
  khác biệt hiếm có thể quan trọng; đây chưa phải causal attribution.

Trace là post hoc trên panel đã mở, không dùng để tái xác nhận v1. Tác dụng
của riêng representation, thiếu dữ liệu skill topology, checkpoint selection
và credit assignment chưa được tách. Model dùng vector N4/K2 và catalogue
21 matching, nên framework này chưa có bằng chứng mở rộng N/K. Không có job
routing, precedence hoặc joint production scheduling.

## Quyết định tiếp theo

Một follow-up có giới hạn sẽ huấn luyện cùng PPO/observation/action/mask,
cùng tổng budget nhưng cho training episode lấy đều từ bốn case. Chọn checkpoint
trên DEV bốn case và đánh giá trên seed mới, kèm v1 checkpoint đóng băng và
ba nonlearning baselines. Nó hỏi liệu **quy trình train + select có bao phủ
nhiều cấu hình** cải thiện cost case-balanced hay không; không tự tách hiệu ứng
train distribution khỏi selection rule. Giữ cost/physics nguyên. Nếu thất bại,
nghiên cứu representation và credit assignment thay vì quét setting tùy ý.
