# Chẩn đoán frozen service-choice: chưa có căn cứ sửa GO-head

Kết quả chẩn đoán **không đạt tiêu chí động cơ cho legal-option GO-head** đã khóa.
Không triển khai hướng sửa đó chỉ vì nó có vẻ hợp lý. Vòng FULL factorization
vẫn không đạt2/16; mục tiêu học từ đầu gầnPPO chưa đạt, scientific tests vẫn đóng.

Protocol3426ac3, codebbe5034c8a6d774c79ec88e1f32a87f87908f48d,6testsPASS rồi
commit/push trước chạy. Artifact:
`technician_capacity_service_choice_diagnostic_v1_20261016T072341887399080Z`.
Đọc144states đã lưu(48states ×3replicas, firstOPEN DEVseed230100000),548source
bindings trước/sau giữ nguyên,0physicalepisodes mới,0updates,0teststates.

576own-policy decisions của candidate/flat/independent/PPO trên cùng states;
288conditional T2decisions của candidate/flat với CÙNG prefix do candidate chọn.
Mỗi policy tự tính latent encoding của mình. Hook giữ nguyên actions/weights và
được gỡ sau đo. Đây là shadow comparison trên phân phối trạng thái của candidate,
không thay thế một đánh giá cost/health độc lập hoặc nhãn hành động tối ưu.

## Tiêu chí đã khóa và kết quả

| Tiêu chí kỹ thuật | Ngưỡng | Quan sát | Đạt? |
|---|---:|---:|---|
| Số slow-only T2contexts |≥12|13|Có|
| Candidate−flat GO-mode rate ở cùng prefix, slow-only |≥0,20|0|Không|
| Candidate−flat long-start fraction ở commonstates |≥0,05|0,051948|Có|

Cả hai policy chọn GO trong13/13slow-only contexts, và58/58fast-available
contexts. P(GO) trên nhóm slow-only trung bình≈0,99705ở candidate và≈0,99983ở
flat. Không có tình huống candidateGO/flatWAIT hoặc ngược lại trong các nhóm này.

Do không đạt ĐỦ ba tiêu chí, giả thuyết cụ thể “candidate nhận việc chậm vì
GO-head không nhìn legal options trong khi flat sẽ WAIT” không được bộ dữ liệu
này ủng hộ. Nó không chứng minh mọi GO-head tốt như nhau, hay không bao giờ cần
thông tin về legal options. Nguồn chỉ có3candidate episodes, context tương quan.

## Quan sát bổ sung về lựa chọn thời lượng

Đếm từ cùng decisionCSV/observation đã bind, không gọi model hay chạy trajectory
thêm. Đây là phân tích mô tả sau đăng ký; không thêm endpoint/alpha.

| Policy trên commonstates | Start1tick | Start2ticks | Start4ticks | Mean committed duration |
|---|---:|---:|---:|---:|
| Candidate |125|13|16|1,396104|
| Matched flat |133|10|11|1,279221|
| Independent |116|14|13|1,370629|
| PPO gốc |111|32|0|1,223776|

Đáng chú ý: PPO có tỷ lệ start>1tick cao hơn candidate trên commonstates,
nhưng không chọn4ticks ở bộ dữ liệu này. Vì vậy ngay cả việc gọi tất cả các
start>1tick là “xấu” cũng không được dữ liệu hỗ trợ. Không suy ra duration penalty.

Ở CÙNG prefixT1 của candidate, candidateT2 luôn chọn dịch vụ ngắn nhất trong
58fast-available và13slow-only contexts; flat có3lần không chọn ngắn nhất ở
nhóm fast. Thất bại không thể được giải thích đơn giản là candidateT2 luôn xếp
hạng chậm hơn flat ở cùng prefix. Lịch phân công ban đầu, trạng thái policy tự
đưa hệ thống vào, và cách học giá trị theo thời gian cần được xem xét tiếp.

Own-policy candidate có9start4ticks ởT1 và7ởT2; flat4ởT1 và7ởT2. Sự khác biệt
trên corpus này tập trung ởT1, không phải chỉ ở GO-mode T2 sau khi giữ chỗ.
Không suy diễn các tỷ lệ này thành thống kê trên toàn bộtest/distribution.

## Quyết định nghiên cứu

Giữ nguyên kết quả âm tính, không mở confirmation hoặc largerN. Không tự động
thêm legal-option pooling, phạt duration, tăng budget hoặc đổi activation. User
cho phép tinh chỉnh có căn cứ; điều đó không biến một giải thích vừa bị đối chiếu
bác bỏ thành lý do cho một vòng train mới.

Bước nghiên cứu tiếp theo phải kiểm tra lựa chọn/lịch chiếm dụng và tín hiệu
return/value trên các trajectory đã có trước khi khóa một follow-up mới. PPO
baseline trong chẩn đoán chỉ là đối chứng đọc; không tạo teacher labels hay
warm-start/dữ liệu huấn luyện. Tất cả kết quả này làDEV/post-hoc, không cóCI.

Có thể lấy artifacts về Mac, không có transfer tự động:

```bash
mkdir -p "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results" &&
rsync -avhP --partial \
  bachng@100.111.83.52:~/ht-pdm-fjsp/artifacts/ \
  "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/"
```
