# Vòng v2: PPO train và chọn checkpoint trên bốn cấu hình

**Verdict đã đăng ký:** H1 không đạt. FULL và audit độc lập đều PASS;
không thay hypothesis hoặc panel sau khi thấy kết quả. Source commit
`526272a52d65a97733550fbaf39292afc665b9bb`. Raw run
`research_result/framework_v2_full_20261019T154912Z/`; audit
`research_result/framework_v2_audit_20261019T160153Z/`.

## Thiết kế và tính toàn vẹn

Giữ kernel request/physical, objective H48+tail16, 56 feature observation,
21 matching catalogue, mask, MaskablePPO MLP [64,64], hyperparameters và
599.040 bước training như v1. Mỗi training episode chọn đều một trong bốn
case; số episode reset/case/seed khoảng 975–1.100, đủ thấy cả bốn cấu hình.
Chọn trong 12 checkpoint theo mean DEV bốn case ×16 seed; checkpoint seed
`63000000`, step 50.176 có DEV mean 934,70. Các checkpoint khác có mean
935,97–944,52; tiến độ thêm không cải thiện chọn lựa bốn case. Đây không
chứng minh optimization không thể học lâu hơn, chỉ giới hạn ngân sách này.

Ubuntu smoke `framework_v2_smoke_20261019T154817Z` hoàn tất 1.024 bước,
8 DEV, 40 eval, checkpoint reload/cost/feasibility PASS. 50 targeted tests
PASS. FULL: 599.040 bước, 768 DEV, 1.280 TEST, 12 checkpoint, 753,0 giây.
Verifier độc lập kiểm tra SHA source/artifact và v1 comparator, seed split,
selection, exact grid, physical seed, reward/cost/tail, request conservation,
mean, paired differences và toàn bộ bootstrap intervals. PASS; zero invalid
assignment. Feasibility đến từ mask và simulator, không phải policy tự học.

## TEST mới, 64 paired physical seed/case

Cost H+tail thấp hơn tốt hơn. Mixture trừ v1 PPO là contrast chính theo case;
CI case-level 95% chỉ mô tả, không hiệu chỉnh multiple testing. Primary là
case-balanced mean của bốn chênh lệch trên từng seed rồi bootstrap 64 seed
nguyên cụm, 20.000 draws.

| Case | FCFS | Risk | Rollout | V1 PPO | Mixture PPO | Mixture − V1 [CI95% mô tả] |
|---|---:|---:|---:|---:|---:|---:|
| Reciprocal scale 2, anchor | 905,59 | 813,66 | 717,69 | 815,06 | 843,08 | +28,02 [13,38; 43,92] |
| Interchangeable scale 1 | 768,83 | 768,73 | 688,73 | 817,27 | 809,12 | −8,14 [−19,66; 2,84] |
| Reciprocal scale 3 | 1.140,73 | 1.017,84 | 1.045,88 | 975,97 | 1.031,94 | +55,97 [25,09; 90,88] |
| Specialist/generalist scale 2 | 1.010,72 | 996,58 | 992,06 | 1.054,25 | 1.012,95 | −41,30 [−55,70; −27,31] |

**Primary pooled mixture − v1 = +8,64** cost/episode; cluster-bootstrap
CI95% [−0,68; +18,91]. Protocol yêu cầu mean âm và CI cận trên âm; cả hai
không đạt. Điểm ước lượng ngược hướng dự đoán, nhưng CI vẫn chứa 0 nên không
khẳng định pooled mixture tệ hơn với mức 95%. Mixture cũng chưa vượt risk ở
specialist: +16,38 [1,75; 30,97], và kém rollout ở anchor: +125,39
[105,17; 145,13]. V1 checkpoint là đối chứng historical đóng băng; vì v2
đổi cả phân phối train lẫn DEV selection, chênh lệch không cô lập một cơ chế.

## Sức khỏe và cơ chế đã mở TEST (exploratory)

Hai trace đầu tiên và hai trace bổ sung theo amendment được giữ nguyên tại
các thư mục `framework_v2_diagnostic_*` timestamped. Amendment thêm metric
matching có thể mở rộng sau khi quan sát mixture ít WAIT rỗng nhưng ít service
starts hơn ở reciprocal. Đây là phân tích post hoc, không sửa verdict.

- Specialist: mixture so v1 giảm WAIT rỗng khi có service hợp lệ từ
  255/3.072 xuống 0/3.072 tick; matching có thể thêm một cặp giảm
  582 xuống 114; service starts tăng 1.733 lên 1.890 trên 64 episode.
  CM queued exposure giảm 22,11 xuống 11,89 machine-tick/episode;
  availability tăng 45,99% lên 49,28%. Cost giảm 41,30 nhưng vẫn cao hơn
  risk và rollout. Mixture chọn slow eligible edge ít hơn trên trace.
- Anchor: mixture cost tăng 28,02; H availability giảm 60,19% xuống 58,03%,
  CM queued exposure tăng 0,69 lên 3,09. Số service starts giảm 2.451 xuống
  2.341 trên các trajectory khác nhau. Chỉ số matching có thể mở rộng giảm,
  nên không thể quy loss đơn giản cho WAIT hoặc thiếu matching maximality.
- Reciprocal scale 3: cost tăng 55,97; availability giảm 50,30% xuống
  47,41%, CM queued exposure tăng 1,41 lên 8,38. Mixture giống risk ở
  97,66% quyết định so với 97,04% của v1. Khác biệt nhỏ trong quyết định
  có thể tác động dài hạn; trace không đủ chứng minh causal mechanism.
- Interchangeable: mixture giảm cost 8,14 nhưng CI mô tả chứa 0; vẫn cao
  hơn risk 40,39 [23,52; 57,92]. Không gọi đây là generalization thành công.

Model selection time trung bình ~0,026–0,027 giây/episode (~0,55 ms/quyết
định) cho hai PPO, risk ~0,0014 giây/episode; rollout ~0,57–0,76 giây/episode
và tốn hàng nghìn simulated transitions. Đây là single-process CPU timing,
không phải latency mạng hoặc end-to-end factory deployment.

## Diễn giải và bước tiếp theo

Training đa cấu hình sửa được một phần lỗi transfer sang hard skill exclusion,
nhưng cùng MLP/catalogue/199.680 bước mỗi seed không giữ được lợi ích timing ở
reciprocal. DEV chọn checkpoint sớm và các score gần nhau; chưa tách được
representation capacity, credit assignment hoặc tradeoff do lựa chọn một
checkpoint chung. RL chưa cạnh tranh nhất quán với risk/rollout. Không đổi cost,
penalty hoặc chỉ giữ specialist để có kết quả dương.

Follow-up có căn cứ là thử policy biểu diễn request–technician edge và
availability/opportunity cost theo cấu trúc, hoặc teacher rollout từ **train
seeds** rồi PPO fine-tune với warm-only/scratch controls, ngân sách teacher và
inference đầy đủ. Trước khi thực hiện phải khóa protocol/seeds/selection/health
và giữ một test panel mới. Không suy teacher giúp hay graph sẽ thắng từ v1/v2.
