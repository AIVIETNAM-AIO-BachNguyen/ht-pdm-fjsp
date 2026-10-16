# Residual service-capacity PPO v1: kết quả âm tính trên N4/K2

**Quyết định: loại thiết kế RSC v1 theo tiêu chí đã khóa.** FULL development
hoàn tất, audit kỹ thuật và kiểm tra độc lập PASS, nhưng chỉ **7/15** tiêu chí
nghiên cứu đạt. Không mở scientific confirmation, không chạy kích thước lớn,
không đổi encoder/LR/horizon hay thêm attention/thành phần để cứu phiên bản này.
Mục tiêu thuật toán học từ đầu gần PPO tập trung vẫn chưa đạt.

Đây là ước lượng trên DEV dùng chọn checkpoint, **không phải kết quả xác nhận
hay tuyên bố có ý nghĩa thống kê**. Kết quả này bác bỏ thiết kế cụ thể trong
benchmark/ngân sách đã đăng ký; không chứng minh mọi biểu diễn capacity đều vô ích.

## Giả thuyết và phần mới được kiểm chứng

Giả thuyết: tóm tắt năng lực phục vụ của technician phía sau giúp technician
hiện tại tránh lấy máy dễ phục vụ chung, để rồi bỏ lại máy ít lựa chọn thay thế.
Giữ PPO clipped chuẩn và học từ trọng số ngẫu nhiên. Đóng góp được đề xuất nằm
ở biểu diễn thông điệp và phối hợp phân công, không ở một objective RL mới.

Với mỗi máy chưa phục vụ/chưa được giữ chỗ và cửa sổ h={1,2,4} ticks, mỗi
technician phía sau chia một đơn vị cơ hội cho các máy họ có thể hoàn thành
một dịch vụ trong cửa sổ đó, tính cả thời gian bận còn lại đã biết. Cộng các
phần này thành ba số cho mỗi máy. Đây là xấp xỉ năng lực một việc/cửa sổ,
không phải matching tối ưu hay dự báo future events.

Actor duyệt technician theo thứ tự cố định, chọn máy hoặc WAIT rồi giữ chỗ.
Ở N4/K2, chỉ technician đầu có technician phía sau; thông điệp của technician
cuối bằng0. Không được suy diễn một cơ chế cập nhật prefix phong phú từ K2.
Policy không liệt kê toàn bộ matching. Mỗi lần sample/replay/greedy dùng K
conditional evaluations; critic chỉ nhận trạng thái vật lý, không nhận hành động
đang đề xuất. Tổng log-probability của các lựa chọn dùng cho tỷ lệ PPO joint.

Năm nhánh cùng thông tin gốc,20.010 tham số, trọng số ban đầu, vai trò RNG,
ngân sách và physical tapes:

| Nhánh | Thông điệp / phối hợp |
|---|---|
| residual_capacity_ppo | Năng lực của technician phía sau; giữ chỗ |
| no_capacity_ppo | Ba kênh bằng0; vẫn giữ chỗ |
| initial_capacity_ppo | Capacity tĩnh của tất cả technician, không xét phần còn lại |
| permuted_capacity_ppo | Giữ các giá trị nhưng gắn sai máy bằng phép dịch vòng cố định |
| independent_capacity_ppo | Không giữ chỗ/prefix; đề xuất độc lập và arbitration khi trùng máy |

MLP thông điệp tăng1.620 tham số, được cấp giống nhau cho mọi nhánh; lớp cuối
khởi tạo0 nên cùng xác suất ban đầu tại cùng prefix hợp lệ. Actor11.423,
critic8.587 tham số. Đây là cùng dung lượng được cấp, không phải khẳng định các
nhánh có cùng lớp hàm hữu hiệu. Toàn bộ trạng thái hiện tại đã có sẵn cho các
nhánh; biểu diễn mới không cấp thêm phép đo.

Các baseline bên ngoài có kiến trúc khác: PPO18.966, localMAPPO15.364,
full-informationMAPPO18.052 tham số. Thắng localMAPPO có ít thông tin hơn không
đủ để kết luận ưu thế. Phối hợp dùng đồ thị hiện tại chung và broker, chưa phải
triển khai hoàn toàn chỉ dùng quan sát cục bộ hay bằng chứng generalization.

## Thiết kế thực nghiệm và tính toàn vẹn

Protocol đăng ký và push ở466a561, trước triển khai. Code được89 kiểm tra xác
nhận rồi commit/push **b44f114a8bc44ca2fefdfcd4eadf18d1e0f5322d** trước FULL.
Chạy trực tiếp trên Ubuntu lab, bỏ riêng smoke theo chỉ dẫn của người dùng.

- 27 fits:9 phương pháp ×3 train seeds mới;491.520 quyết định/fit.
- 13.271.040 quyết định,276.480 train episodes;7.680 optimizer steps/fit.
- Vector16, rollout3.072, Adam3e-4,4epochs, minibatch256; PPO clip0,2.
- DEV32 seeds ghép cặp,5checkpoints/fit:4.320 DEV episodes và32 deadline reference.
- Tất cả fits chạy đủ; chọn checkpoint có mean DEV cost nhỏ nhất, tie lấy sớm nhất.
- Giữ cả3replicas của mọi nhánh. Candidate chọn491.520 ở cả3replicas.
- 8processes,torch1,tqdm; elapsed1.556,015s, khoảng25phút56giây.
- Artifact mới, source/registry/config/physics/checkpoint/output bindings đầy đủ.

Các dòng trên liệt kê số liệu; seed cụ thể và toàn bộ15ngưỡng nằm trong
[protocol](technician_residual_service_capacity_ppo_v1_plan.md). DEVtrain
220000000..2; DEVphysical220100000..31.366raw và82.429derived seeds được audit
không trùng phạm vi đã đăng ký. Tập confirmation mới vẫn đóng; round2alpha.05/6
chưa được dùng. Nếu DEV đạt mới được xác nhận5replicas, TEST256 và bootstrap
200.000draws với Bonferroni15; vòng này không đủ điều kiện đi tiếp.

## Kết quả đầy đủ trên DEV

| Phương pháp | Cost/máy | Availability | Pending/máy | Failures/máy |
|---|---:|---:|---:|---:|
| RSC candidate |269,132813|0,667426|0,007813|0,390625|
| Không thông điệp |272,674479|0,664551|0,002604|0,658854|
| Capacity tĩnh |269,023438|0,669108|0,002604|0,533854|
| Thông điệp gắn sai máy |271,166667|0,667969|0,005208|0,609375|
| Độc lập |269,398438|0,670790|0|0,570313|
| PPO gốc |260,713542|0,668891|0|0,080729|
| PPO normalized |260,445313|0,669108|0|0,070313|
| Local MAPPO |269,153646|0,669108|0|0,528646|
| Full-information MAPPO |263,609375|0,667914|0|0,307292|
| Deadline reference |280,320313|0,665365|0,015625|0,992188|

Gain=1−cost_candidate/cost_reference; dương nghĩa candidate rẻ hơn.

| So sánh cost | Gain | Ngưỡng DEV | Kết quả |
|---|---:|---:|---|
| vs không thông điệp |+1,2989%|≥+0,5%|Đạt|
| vs capacity tĩnh |−0,0407%|≥+0,5%|Không đạt|
| vs thông điệp gắn sai máy |+0,7500%|≥+0,5%|Đạt|
| vs độc lập |+0,0986%|≥+0,5%|Không đạt|
| vs PPO gốc |−3,2293%|≥−1%|Không đạt|
| vs PPO normalized |−3,3356%|≥−1%|Không đạt|
| vs localMAPPO |+0,007740%|>0|Đạt ở point screen, chưa có xác nhận|
| vs full-informationMAPPO |−2,0953%|>0|Không đạt|

Availability và pending đạt giới hạn với cả hai PPO. Nhưng failures/máy cao
hơn PPOgốc0,309896 và PPOnormalized0,320313, vượt ngưỡng+0,25. Cost tốt hơn
một ablation không thể bù cho thất bại này. Zero invalid assignments; actor-only
p95=2,215717ms trên432samples cô lập sau khi training pool đóng, đạt≤50ms.
Thời gian này không gồm dựng observation, physics hay truyền mạng, và N4 không
chứng minh được khả năng mở rộng.

## Bằng chứng về cơ chế: không đạt kỳ vọng

16 trạng thái cố định có M1/M2 cùng tuổi/trạng thái failure, T1 phục vụ cả hai
trong1tick; T2 cần2ticks ở M1 nhưng4ticks ở M2. M3/M4 không urgent. Trong cửa
sổ2ticks, M2 ít lựa chọn thay thế hơn; phép đo là xác suất T1 chọn M2. Các trạng
thái này không dùng cho huấn luyện hay chọn checkpoint.

| Replica | Candidate P(M2) | Không thông điệp P(M2) | Candidate khi zero-message cùng trọng số |
|---|---:|---:|---:|
|220000000|0,500371|0,499678|0,499994|
|220000001|0,500020|0,499326|0,499993|
|220000002|0,525819|0,499020|0,499991|

Candidate mean0,508737; đối chứng0,499341; khác biệt **+0,9395 điểm phần trăm**,
không đạt ngưỡng **+5điểm** đã khóa. Chỉ replica thứ3 thể hiện thay đổi đáng thấy
trong probe; không được chọn riêng nó để tuyên bố cơ chế có hiệu quả.

Phân tích phụ đã ghi rõ sau đăng ký, trong lúc DEV còn chạy: xác suất M2 khi
đã chọn phục vụ thay vì WAIT cũng chỉ khác khoảng0,93điểm phần trăm. WAIT vốn
rất thấp ở cả hai nhánh. Phân tích này không thay tiêu chí chính, không thêm
trajectory và không cứu được kết quả. Nó chỉ kiểm tra một cách giải thích khác.

Bản độc lập đạt xác suất critical-first khoảng0,8142 trên probe nhưng cost/health
vẫn không đạt mục tiêu. Vì vậy probe không phải nhãn hành động tối ưu hay thước
đo thay thế cho kết quả vật lý. Same-weight knockout chỉ là kiểm tra nhạy cảm
ngoài phân phối training, không phải một ablation được huấn luyện độc lập.

Candidate không có tranh chấp nhờ giữ chỗ. Bản độc lập có trung bình7,8125
tranh chấp và đề xuất bị loại/episode ở checkpoint DEV đã chọn; localMAPPO
3,6146tranh chấp và fullMAPPO1,2396. Đây là phân tích mô tả các trajectory sẵn có.
Dù giảm tranh chấp, candidate chỉ giảm cost0,0986% so với bản độc lập, dưới
ngưỡng0,5%. Chưa chứng minh được lợi ích cost cần thiết của cơ chế phối hợp này.

## Điều được kết luận và giới hạn

Lợi ích1,30% so với nhánh zero-message là một quan sát phát triển đáng giữ.
Tuy nhiên, capacity tĩnh hơi tốt hơn và nhánh độc lập gần ngang cost; dữ liệu
chưa hỗ trợ lợi ích riêng của cách biểu diễn năng lực *còn lại theo thứ tự*.
Không được rút gọn kết quả thành “thông điệp đã hiệu quả” rồi bỏ các đối chứng
bất lợi, health hoặc điều kiện gần PPO.

Quyết định đã khóa là loại RSCv1 trong cấu hình/ngân sách này. Không có
confirmation CI, tuyên bố vượt PPO, thành công N4, generalization hay scalability.
Không đổi horizon/LR/encoder, thêm attention, đổi tên cơ chế hay mở test panel để
cứu phiên bản. Một giả thuyết khác cần động cơ độc lập và protocol mới; các kết
quả âm tính này phải được giữ nguyên.

Các vòng scarcity trước dùng giá cơ hội học được; completion dùng lịch/credit;
marginal-matching dùng likelihood xác suất matching vật lý. Chúng cũng chưa
đạt mục tiêu. Diagnostic decoder của vòng marginal đã chạy đủ nhưng source
FAILED ở kiểm tra float64-vs-float32; adjudication chỉ đọc xác nhận tính nhất quán
float32, vẫn giữ source FAILED và thấy exact decoder không cải thiện candidate.
Không gộp vòng diagnostic đó thành một xác nhận thành công hay bỏ lịch sử thất bại.

## Nguồn, kiểm tra độc lập và tái lập

FULL artifact:
`artifacts/technician_residual_service_capacity_ppo_v1_development_20261016T032534946293781Z`.

Báo cáo độc lập:
`artifacts/technician_residual_service_capacity_ppo_v1_development_report_20261016T035147990511794Z`.

Verifier kiểm tra519output bindings, source seals, toàn bộ grid/budgets/tapes/
reward/checkpoint selection,15point gates và latency. Một phép tính NumPy/float64
độc lập dựng lại240xác suất probe và48xác suất knockout: sai số tối đa3,783e-7
và3,385e-7, trong dung sai xác suất2e-6. Điều này chứng minh phép tính nhất quán,
không chứng minh cơ chế tốt. Hai PNG đã được kiểm tra trực quan; min–max/dots
là3training replicas, không phải confidence intervals.

Code và protocol được push trước FULL. Chạy lại nguyên command sẽ tạo directory
mới, nhưng kết quả âm tính không tự cho phép lặp lại full run hay tái sử dụng
scientific test. Không SSH, không tự chuyển artifact. Có thể lấy kết quả về Mac:

```bash
mkdir -p "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results" &&
rsync -avhP --partial \
  bachng@100.111.83.52:~/ht-pdm-fjsp/artifacts/ \
  "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/"
```
