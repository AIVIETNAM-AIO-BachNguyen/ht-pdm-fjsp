# Tách service mass và xếp hạng máy: kết quả FULL N4/K2

**Vòng phát triển không đạt:2/16tiêu chí.** Tinh chỉnh làm policy phản ứng rõ
hơn với capacity trên probe, nhưng cost và sức khỏe hệ thống xấu hơn. Không mở
tập xác nhận, không mở rộng N. Mục tiêu thuật toán học từ đầu vẫn chưa đạt.

Kết quả cũ RSCv1 vẫn là7/15không đạt. Người dùng cho phép tinh chỉnh cần thiết;
vòng này kiểm tra một nguyên nhân cụ thể, không thay đổi phán quyết/threshold
của vòng cũ và không có teacher, distillation hay warm-start từ PPO.

## Phép thử và đối chứng

Chẩn đoán trước đó thấy bid-head bão hòa và làm mất khác biệt capacity ở một số
checkpoint. Giả thuyết: flat softmax dùng cùng machine scores cho cả việc tăng
xác suất phục vụ và chọn máy; tách hai phần có thể giữ được thông tin xếp hạng.

Với c máy hợp lệ, dùng P(WAIT)=sigmoid(w−log(c)), còn P(máy)=P(GO)*softmax(bids).
Không có lựa chọn hợp lệ thì forcedWAIT. Mô hình vẫn dùng objective PPO joint
chuẩn và entropy của categorical đầy đủ, gamma1, MC returns, cùng LR3e-4,
491.520quyết định/fit. Greedy chọn xác suất lớn nhất trên TOÀN BỘ token, không
đổi thành greedy tách cổng nhị phân. Không thêm encoder, attention hay tham số.

Sáu core arms đều20.010tham số, cùng tensor ngẫu nhiên, cùng thông tin gốc và
physical/RNG roles: capacity thật, zero, capacity tĩnh, capacity gắn sai máy,
độc lập không giữ chỗ, và flat capacity control. Bid/WAIT output heads khởi tạo0
ở TẤT CẢ core arms, tạo cùng xác suất vật lý ban đầu ở cùng prefix hợp lệ.
Flat control là so sánh trong vòng cần thiết: không được gán chênh lệch với
RSCv1 cũ cho factorization khi seed và khởi tạo head của hai vòng khác nhau.

Generic policy factorization có prior art; xem
[protocol và đối chiếu](technician_residual_capacity_factorization_v1_plan.md).
Đây không phải objective PPO mới hay phát minh hierarchical RL. Giả thuyết cụ
thể là parameterization này giúp học được phối hợp capacity có lợi ở bài toán
maintenance, điều phải kiểm chứng bằng cost/health và ablation.

## Tính toàn vẹn và quy mô vòng chạy

Protocol commit a89d3b1 trước triển khai.99kiểm tra PASS trước code commit/push
**a10e049e3f8b4e242efbc879b2d3dc3c13925bf4** và FULL trực tiếp trên lab.
Không chạy riêng smoke theo user override.

- 30fits:10phương pháp ×3replicas từ khởi tạo ngẫu nhiên.
- 14.745.600quyết định;307.200train episodes;7.680optimizer steps/fit.
- 4.800DEV episodes:5checkpoints ×32physicalseeds ×30fits, cộng32deadline.
- Tất cả chạy đủ491.520quyết định; checkpoint min meanDEVcost, tie lấy sớm nhất.
- Candidate chọn368640/122880/245760 theo thứ tự3trainseeds; giữ đủ cả3.
- Elapsed1.695,832s≈28phút16giây;8workers,torch1,tqdm;0invalids.
- Actor-only p95=2,760563ms trên432samples cô lập, đạt giới hạn50ms.

DEVtrain230000000..2,DEVphysical230100000..31.366raw và82.429derived seeds được
audit với lịch sử. Tập confirmation mới vẫn đóng, round2alpha.05/6chưa dùng.
Không có scientific CI hoặc tuyên bố vượt baseline trên test chưa thấy.

## Kết quả DEV tại checkpoint được chọn

| Phương pháp | Cost/máy | Availability | Pending/máy | Policy failures/máy |
|---|---:|---:|---:|---:|
| Factored capacity candidate |305,354167|0,638021|0,028646|1,575521|
| Factored zero capacity |301,367188|0,652018|0,010417|1,736979|
| Factored capacity tĩnh |297,700521|0,643121|0,013021|1,364583|
| Factored capacity sai máy |310,989583|0,625705|0,020833|1,544271|
| Factored độc lập |275,486979|0,669596|0,005208|0,882813|
| Matched flat capacity |271,916667|0,670410|0,007813|0,677083|
| PPO gốc |261,765625|0,669162|0|0,138021|
| PPO normalized |261,742188|0,668566|0|0,166667|
| Local MAPPO |263,513021|0,670356|0|0,281250|
| Full-information MAPPO |265,809896|0,666992|0,002604|0,369792|
| Deadline reference |279,914063|0,668457|0,023438|1,062500|

Candidate cost cao hơn flat12,2970%, cao hơn PPOgốc16,6517% và PPOnormalized
16,6622%; không đạt biên gầnPPO1%. Nó cũng cao hơn bản zero1,3230%, bản tĩnh
2,5709%, bản độc lập10,8416%, localMAPPO15,8782% và fullMAPPO14,8769%.
Chỉ gain1,8121%so với capacity gắn sai máy vượt ngưỡng cost0,5%.

Cả6health endpoints so với haiPPO đều không đạt. So với PPOgốc: availability
−0,031141(vượt giới hạn−0,01); pending/máy+0,028646(vượt+0,025);
failures/máy+1,437500(vượt+0,25). Không nới giới hạn vì probe đẹp hơn.

## Cơ chế đo được: cải thiện probe nhưng không đạt lợi ích vận hành

Trên16trạng thái DEV đã khóa, T1 phục vụ M1/M2 cùng1tick, T2 cần2ticks choM1
và4choM2; hai máy cùng tuổi/failure, các máy khác không urgent. Đây là probe
về cơ hội thay thế, không phải nhãn hành động tối ưu hay dữ liệu huấn luyện.

Candidate P(chọnM2)=0,847003, zero=0,455954: khác biệt **+39,1049điểm phần trăm**,
đạt ngưỡng+5điểm. Điều kiện đã chọn phục vụ, khác biệt còn **+25,3582điểm**.
Vì vậy một phần khác biệt đến từ WAIT, nhưng không phải toàn bộ.

Ở checkpoint được chọn, bid-layer2saturation trung bình38,6188%ở candidate,
59,4136%ở flat; hiddencontrastM1/M2 là4,4389so với0,5577. Các phép đo này chỉ
mô tả trên probe16states và checkpoint khác nhau do chọn bằng cost. Không được
suy thành giảm bão hòa trên mọi trạng thái hoặc bằng chứng học luôn tốt hơn.

Chính đối chứng flat chứng minh sự khác nhau giữa một sửa đổi có tác dụng lên
biểu diễn và một thuật toán có ích: candidate thể hiện cơ chế rõ hơn nhưng trả
cost cao hơn, hoàn thành ít công việc hơn và làm hệ thống kém khỏe hơn. Chỉ đạt
hai tiêu chí về probe và capacity-sai-máy không đủ đi scientific confirmation.

## Chênh lệch cost nằm ở đâu?

Phân tích chỉ đọc sau đăng ký dùng96DEVepisodes/phương pháp tại checkpoint đã
chọn. Các cost components cộng lại đúng extendedcost ở TỪNG episode.

Candidate−flat=**33,4375cost/máy**, phân rã chính xác:

| Thành phần | Chênh lệch/máy |
|---|---:|
| Unavailability |+18,156250|
| Failure cost |+16,210938|
| Maintenance |−0,929688|
| Waiting penalty |0|

Candidate trung bình hoàn thành69,8646jobs/episode, flat77,7917. Tỷ lệ service
ticks quan sát trên mỗi job bắt đầu≈1,2056so với≈1,0201. Tỷ lệ này có thể bị
censor ở cuối episode, nên không gọi nó là thời lượng chuẩn xác của mọi job.
Nó gợi ý cần kiểm tra quyết định nhận những dịch vụ chiếm dụng lâu, không chứng
minh rằng phạt duration sẽ sửa được vấn đề. Các chẩn đoán duration cũ vẫn phải
được giữ trong hồ sơ; không lặp lại duration penalty chỉ từ tương quan này.

Candidate không chọn cuối budget ở mọi replica và đường học sau122880gần phẳng.
Không có bằng chứng đủ để lập tức nhân đôi thời gian huấn luyện. Bước tiếp theo
phù hợp là đọc các trạng thái/conditional đã lưu, so sánh GO/WAIT và tập lựa chọn
còn hợp lệ ở cùng prefix, trước khi đăng ký một thay đổi tiếp theo.

## Kiểm tra độc lập và artifact

FULL source:
`artifacts/technician_residual_capacity_factorization_v1_development_20261016T064203318666531Z`.

Báo cáo:
`artifacts/technician_capacity_factorization_v1_development_report_20261016T071155485329490Z`.

Verifier đối chiếu548output bindings, source/protocol/registry với blob của
commit đã push, tính lại tensor initialization và zeroheads, toàn bộ budgets,
physicaltapes/reward/checkpointselection,16gates và latency. NumPy/float64 độc
lập dựng lại288vector-probe và48knockout probabilities; maxabsoluteerrors
3,021e-7và1,652e-7, nhỏ hơn dung sai2e-6; kiểm tra cả saturation/contrast.
Ba PNG đã được kiểm tra trực quan. Min–max/dots là3replicas, không phảiCI.

Lấy artifact về Mac bằng lệnh dưới đây; không có transfer nào được tự khởi chạy:

```bash
mkdir -p "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results" &&
rsync -avhP --partial \
  bachng@100.111.83.52:~/ht-pdm-fjsp/artifacts/ \
  "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/"
```
